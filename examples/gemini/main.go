package main

import (
	"context"
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== Google Gemini Complete Example ===")

	client := litellm.New(litellm.WithGemini("your-gemini-key"))

	// Basic conversation
	fmt.Println("\n--- Basic Chat ---")
	response, err := client.Complete(context.Background(), &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain the difference between AI, ML, and Deep Learning"},
		},
		MaxTokens:   litellm.IntPtr(200),
		Temperature: litellm.Float64Ptr(0.6),
	})
	if err != nil {
		log.Fatalf("Basic chat failed: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)

	// Streaming conversation
	fmt.Println("\n--- Streaming Chat ---")
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a detailed explanation of how neural networks work"},
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Streaming failed: %v", err)
	}
	defer stream.Close()

	fmt.Print("Streaming explanation: ")
	for {
		chunk, err := stream.Read()
		if err != nil {
			log.Fatalf("Stream read failed: %v", err)
		}

		if chunk.Done {
			break
		}

		if chunk.Type == litellm.ChunkTypeContent {
			fmt.Print(chunk.Content)
		}
	}
	fmt.Println()

	// Function calling
	fmt.Println("\n--- Function Calling ---")
	tools := []litellm.Tool{
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "search_knowledge",
				Description: "Search for information in a knowledge base",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "Search query",
						},
						"category": map[string]interface{}{
							"type": "string",
							"enum": []string{"science", "technology", "history", "general"},
						},
					},
					"required": []string{"query"},
				},
			},
		},
	}

	toolResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "gemini-2.5-pro",
		Messages: []litellm.Message{
			{Role: "user", Content: "Search for information about quantum computing"},
		},
		Tools:      tools,
		ToolChoice: "auto",
	})
	if err != nil {
		log.Fatalf("Function calling failed: %v", err)
	}

	if len(toolResp.ToolCalls) > 0 {
		fmt.Printf("🔧 Tool called: %s\n", toolResp.ToolCalls[0].Function.Name)
		fmt.Printf("📋 Arguments: %s\n", toolResp.ToolCalls[0].Function.Arguments)
	} else {
		fmt.Printf("Response: %s\n", toolResp.Content)
	}

	// Large context processing
	fmt.Println("\n--- Large Context Processing ---")
	longText := `
	Artificial Intelligence (AI) is a broad field of computer science focused on creating systems
	capable of performing tasks that typically require human intelligence. Machine Learning (ML)
	is a subset of AI that enables computers to learn and improve from experience without being
	explicitly programmed. Deep Learning is a subset of ML that uses neural networks with multiple
	layers to model and understand complex patterns in data.

	The history of AI dates back to the 1950s when Alan Turing proposed the Turing Test. Since then,
	AI has evolved through various phases including expert systems, machine learning, and now deep
	learning. Modern AI applications include natural language processing, computer vision, robotics,
	and autonomous systems.

	Current challenges in AI include explainability, bias, privacy, and the need for large amounts
	of training data. Future directions include artificial general intelligence (AGI), quantum
	machine learning, and neuromorphic computing.
	`

	contextResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "gemini-2.5-pro",
		Messages: []litellm.Message{
			{Role: "user", Content: fmt.Sprintf("Summarize this text in 3 key points:\n\n%s", longText)},
		},
		MaxTokens:   litellm.IntPtr(150),
		Temperature: litellm.Float64Ptr(0.3),
	})
	if err != nil {
		log.Fatalf("Large context processing failed: %v", err)
	}

	fmt.Printf("Summary: %s\n", contextResp.Content)
}
