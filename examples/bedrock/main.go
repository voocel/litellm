package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/voocel/litellm"
)

func main() {
	// Auto-discovery from environment variables
	// Set these environment variables:
	//   AWS_ACCESS_KEY_ID=your-access-key-id
	//   AWS_SECRET_ACCESS_KEY=your-secret-access-key
	//   AWS_REGION=us-east-1 (optional, defaults to us-east-1)
	//   AWS_SESSION_TOKEN=your-session-token (optional, for temporary credentials)

	// Explicit configuration
	client, err := litellm.New(
		litellm.WithBedrock(
			os.Getenv("AWS_ACCESS_KEY_ID"),
			os.Getenv("AWS_SECRET_ACCESS_KEY"),
			os.Getenv("AWS_REGION"), // optional, defaults to "us-east-1"
		),
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Basic chat with Claude on Bedrock
	fmt.Println("=== Example 1: Basic Chat ===")
	response, err := client.Chat(ctx, &litellm.Request{
		Model: "anthropic.claude-sonnet-4-5-20250929-v1:0",
		Messages: []litellm.Message{
			{Role: "user", Content: "What is the capital of France? Answer in one sentence."},
		},
	})
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d input, %d output tokens\n\n", response.Usage.PromptTokens, response.Usage.CompletionTokens)

	// Chat with Amazon Nova
	fmt.Println("=== Example 2: Amazon Nova ===")
	response, err = client.Chat(ctx, &litellm.Request{
		Model: "amazon.nova-lite-v1:0",
		Messages: []litellm.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Explain quantum computing in 2 sentences."},
		},
		MaxTokens: litellm.IntPtr(200),
	})
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d input, %d output tokens\n\n", response.Usage.PromptTokens, response.Usage.CompletionTokens)

	// Streaming response
	fmt.Println("=== Example 3: Streaming ===")
	stream, err := client.Stream(ctx, &litellm.Request{
		Model: "anthropic.claude-haiku-4-5-20251001-v1:0",
		Messages: []litellm.Message{
			{Role: "user", Content: "Count from 1 to 5, explaining each number."},
		},
		MaxTokens: litellm.IntPtr(300),
	})
	if err != nil {
		log.Fatalf("Stream failed: %v", err)
	}
	defer stream.Close()

	fmt.Print("Streaming: ")
	for {
		chunk, err := stream.Next()
		if err != nil {
			log.Fatalf("Stream read failed: %v", err)
		}
		if chunk.Done {
			break
		}
		if chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	}
	fmt.Println()

	// Tool calling (function calling)
	fmt.Println("=== Example 4: Tool Calling ===")
	response, err = client.Chat(ctx, &litellm.Request{
		Model: "anthropic.claude-sonnet-4-5-20250929-v1:0",
		Messages: []litellm.Message{
			{Role: "user", Content: "What's the weather in Tokyo?"},
		},
		Tools: []litellm.Tool{
			{
				Type: "function",
				Function: litellm.FunctionDef{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "The city name",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
	})
	if err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	if len(response.ToolCalls) > 0 {
		fmt.Printf("Tool called: %s\n", response.ToolCalls[0].Function.Name)
		fmt.Printf("Arguments: %s\n", response.ToolCalls[0].Function.Arguments)
	} else {
		fmt.Printf("Response: %s\n", response.Content)
	}
}
