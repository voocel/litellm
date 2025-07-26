package main

import (
	"context"
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== Anthropic Claude Complete Example ===")

	client := litellm.New(litellm.WithAnthropic("your-anthropic-key"))

	// Basic conversation
	fmt.Println("\n--- Basic Chat ---")
	response, err := client.Complete(context.Background(), &litellm.Request{
		Model: "claude-4-sonnet",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain the concept of recursion in programming"},
		},
		MaxTokens:   litellm.IntPtr(200),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Basic chat failed: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)

	// Streaming conversation
	fmt.Println("\n--- Streaming Chat ---")
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "claude-4-sonnet",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a creative story about a robot learning to paint"},
		},
		MaxTokens:   litellm.IntPtr(300),
		Temperature: litellm.Float64Ptr(0.9),
	})
	if err != nil {
		log.Fatalf("Streaming failed: %v", err)
	}
	defer stream.Close()

	fmt.Print("Streaming story: ")
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
				Name:        "calculate",
				Description: "Perform mathematical calculations",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expression": map[string]interface{}{
							"type":        "string",
							"description": "Mathematical expression to evaluate",
						},
						"operation": map[string]interface{}{
							"type": "string",
							"enum": []string{"add", "subtract", "multiply", "divide"},
						},
					},
					"required": []string{"expression"},
				},
			},
		},
	}

	toolResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "claude-4-sonnet",
		Messages: []litellm.Message{
			{Role: "user", Content: "Calculate 156 + 789 for me"},
		},
		Tools:      tools,
		ToolChoice: "auto",
	})
	if err != nil {
		log.Fatalf("Function calling failed: %v", err)
	}

	if len(toolResp.ToolCalls) > 0 {
		fmt.Printf("Tool called: %s\n", toolResp.ToolCalls[0].Function.Name)
		fmt.Printf("Arguments: %s\n", toolResp.ToolCalls[0].Function.Arguments)
	} else {
		fmt.Printf("Response: %s\n", toolResp.Content)
	}

	// Multi-turn conversation
	fmt.Println("\n--- Multi-turn Conversation ---")
	messages := []litellm.Message{
		{Role: "user", Content: "What is machine learning?"},
	}

	// First round
	resp1, err := client.Complete(context.Background(), &litellm.Request{
		Model:       "claude-4-sonnet",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(100),
		Temperature: litellm.Float64Ptr(0.5),
	})
	if err != nil {
		log.Printf("Multi-turn round 1 failed: %v", err)
		return
	}

	fmt.Printf("Claude: %s\n", resp1.Content)

	// Add assistant response to conversation history
	messages = append(messages, litellm.Message{
		Role:    "assistant",
		Content: resp1.Content,
	})

	// Second round
	messages = append(messages, litellm.Message{
		Role:    "user",
		Content: "Can you give me a simple example?",
	})

	resp2, err := client.Complete(context.Background(), &litellm.Request{
		Model:       "claude-4-sonnet",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(150),
		Temperature: litellm.Float64Ptr(0.5),
	})
	if err != nil {
		log.Printf("Multi-turn round 2 failed: %v", err)
		return
	}

	fmt.Printf("Claude: %s\n", resp2.Content)
}
