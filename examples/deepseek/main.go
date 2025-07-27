package main

import (
	"context"
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== DeepSeek Complete Example ===")

	// Method 1: Environment variable
	// export DEEPSEEK_API_KEY="your-deepseek-api-key"
	//client := litellm.New()

	// Method 2: Manual configuration
	client := litellm.New(litellm.WithDeepSeek("sk-xxx"))

	// Basic conversation
	fmt.Println("\n--- Basic Chat ---")
	response, err := client.Complete(context.Background(), &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain the concept of artificial intelligence"},
		},
		MaxTokens:   litellm.IntPtr(200),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Basic chat failed: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)

	// Reasoning model - DeepSeek's signature feature
	fmt.Println("\n--- Reasoning Model (DeepSeek-R1) ---")
	reasoningResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "deepseek-reasoner",
		Messages: []litellm.Message{
			{Role: "user", Content: "who are you?"},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.1),
	})
	if err != nil {
		log.Fatalf("Reasoning failed: %v", err)
	}

	fmt.Printf("Answer: %s\n", reasoningResp.Content)
	if reasoningResp.Reasoning != nil {
		fmt.Printf("Reasoning process: %s\n", reasoningResp.Reasoning.Content)
	}

	// Streaming conversation
	fmt.Println("\n--- Streaming Chat ---")
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "deepseek-reasoner",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a creative story about a programmer who discovers AI consciousness"},
		},
		MaxTokens:   litellm.IntPtr(300),
		Temperature: litellm.Float64Ptr(0.8),
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
		} else if chunk.Type == litellm.ChunkTypeReasoning {
			// DeepSeek reasoning model's streaming reasoning process
			fmt.Printf("[Thinking: %s]", chunk.Reasoning.Content)
		}
	}
	fmt.Println()

	// Function calling
	fmt.Println("\n--- Function Calling ---")
	tools := []litellm.Tool{
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "code_analyzer",
				Description: "Analyze code for bugs and improvements",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"code": map[string]interface{}{
							"type":        "string",
							"description": "Code to analyze",
						},
						"language": map[string]interface{}{
							"type": "string",
							"enum": []string{"python", "javascript", "go", "java"},
						},
					},
					"required": []string{"code", "language"},
				},
			},
		},
	}

	toolResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{Role: "user", Content: "Analyze this Python code: def factorial(n): return n * factorial(n-1)"},
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

	// Advanced parameters
	fmt.Println("\n--- Advanced Parameters ---")
	advancedResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{Role: "user", Content: "Generate a unique and creative product name for an AI-powered coding assistant"},
		},
		MaxTokens:   litellm.IntPtr(100),
		Temperature: litellm.Float64Ptr(0.9),
		Extra: map[string]interface{}{
			"top_p":             0.95,
			"frequency_penalty": 0.2,
			"presence_penalty":  0.1,
		},
	})
	if err != nil {
		log.Fatalf("Advanced parameters failed: %v", err)
	}

	fmt.Printf("Creative response: %s\n", advancedResp.Content)
}
