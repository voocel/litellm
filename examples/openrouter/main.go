package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== OpenRouter Complete Example ===")

	// Get API key from environment variable
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("Please set OPENROUTER_API_KEY environment variable")
	}

	// Initialize client with OpenRouter provider
	client := litellm.New(litellm.WithOpenRouter(apiKey))

	// Basic conversation
	fmt.Println("\n--- Basic Chat ---")
	response, err := client.Complete(context.Background(), &litellm.Request{
		Model: "anthropic/claude-3.5-sonnet",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain quantum computing in simple terms"},
		},
		MaxTokens:   litellm.IntPtr(150),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Basic chat failed: %v", err)
	}
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %+v\n", response.Usage)

	// Reasoning model example
	fmt.Println("\n--- Reasoning Model ---")
	reasoningResponse, err := client.Complete(context.Background(), &litellm.Request{
		Model: "anthropic/claude-3.7-sonnet",
		Messages: []litellm.Message{
			{Role: "user", Content: "Solve this step by step: What is 15 * 23 + 7?"},
		},
		MaxTokens:        litellm.IntPtr(2048),
		Temperature:      litellm.Float64Ptr(0.1),
		ReasoningEffort:  "medium",
		ReasoningSummary: "Show your work",
		UseResponsesAPI:  true,
	})
	if err != nil {
		log.Fatalf("Reasoning request failed: %v", err)
	}
	fmt.Printf("Response: %s\n", reasoningResponse.Content)
	if reasoningResponse.Reasoning != nil {
		fmt.Printf("Reasoning: %s\n", reasoningResponse.Reasoning.Summary)
	}
	fmt.Printf("Usage: %+v\n", reasoningResponse.Usage)

	// Streaming example
	fmt.Println("\n--- Streaming Chat ---")
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "openai/gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "Count from 1 to 5, one number per line."},
		},
		MaxTokens:   litellm.IntPtr(100),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Stream request failed: %v", err)
	}
	defer stream.Close()

	fmt.Print("Streaming response: ")
	for {
		chunk, err := stream.Read()
		if err != nil {
			log.Fatalf("Stream read failed: %v", err)
		}

		if chunk.Done {
			break
		}

		if chunk.Type == litellm.ChunkTypeContent && chunk.Content != "" {
			fmt.Print(chunk.Content)
		}

		if chunk.Type == litellm.ChunkTypeReasoning && chunk.Reasoning != nil {
			fmt.Printf("[Reasoning: %s]", chunk.Reasoning.Summary)
		}
	}
	fmt.Println("\n--- Streaming Complete ---")

	// Multiple models example
	fmt.Println("\n--- Multiple Models ---")
	models := []string{
		"openai/gpt-4o-mini",
		"anthropic/claude-3.5-sonnet",
		"google/gemini-pro-1.5",
	}

	for _, model := range models {
		fmt.Printf("\nTesting model: %s\n", model)
		response, err := client.Complete(context.Background(), &litellm.Request{
			Model: model,
			Messages: []litellm.Message{
				{Role: "user", Content: "Say hello and tell me your name."},
			},
			MaxTokens:   litellm.IntPtr(50),
			Temperature: litellm.Float64Ptr(0.7),
		})
		if err != nil {
			fmt.Printf("Error with %s: %v\n", model, err)
			continue
		}
		fmt.Printf("Response: %s\n", response.Content)
	}

	fmt.Println("\n=== OpenRouter Example Complete ===")
}
