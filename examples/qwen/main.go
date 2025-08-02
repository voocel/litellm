package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== Qwen (DashScope) Complete Example ===")

	// Method 1: Environment variable
	// export DASHSCOPE_API_KEY="your-dashscope-api-key"
	//client := litellm.New()

	// Method 2: Manual configuration
	client := litellm.New(litellm.WithQwen(os.Getenv("QWEN_API_KEY"), os.Getenv("QWEN_BASE_URL")))

	// Basic conversation with Qwen3-Coder-Plus
	fmt.Println("\n--- Basic Chat with Qwen3-Coder-Plus ---")
	response, err := client.Complete(context.Background(), &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{Role: "system", Content: "You are a helpful coding assistant."},
			{Role: "user", Content: "Write a Python function to calculate the factorial of a number using recursion"},
		},
		MaxTokens:   litellm.IntPtr(300),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Basic chat failed: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)

	// Code generation with different models
	fmt.Println("\n--- Code Generation with Qwen3-Coder-Flash ---")
	codeResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "qwen3-coder-flash",
		Messages: []litellm.Message{
			{Role: "user", Content: "Create a Go function that implements a binary search algorithm. Include comments explaining the logic."},
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.3),
	})
	if err != nil {
		log.Fatalf("Code generation failed: %v", err)
	}

	fmt.Printf("Generated code: %s\n", codeResp.Content)

	// Streaming conversation
	fmt.Println("\n--- Streaming Chat ---")
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain the differences between synchronous and asynchronous programming in JavaScript with examples"},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.8),
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

		fmt.Print(chunk.Content)
	}
	fmt.Println()

	// Function calling - Qwen supports tool calling
	fmt.Println("\n--- Function Calling ---")
	tools := []litellm.Tool{
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "write_file",
				Description: "Write content to a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "The file path to write to",
						},
						"content": map[string]interface{}{
							"type":        "string",
							"description": "The content to write to the file",
						},
					},
					"required": []string{"path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "read_file",
				Description: "Read content from a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "The file path to read from",
						},
					},
					"required": []string{"path"},
				},
			},
		},
	}

	toolResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{Role: "user", Content: "Create a simple Python hello world program and save it to hello.py"},
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
		{Role: "user", Content: "I need help with a Python web scraping project"},
	}

	// First turn
	resp1, err := client.Complete(context.Background(), &litellm.Request{
		Model:       "qwen3-coder-plus",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(200),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Multi-turn conversation failed: %v", err)
	}

	fmt.Printf("Assistant: %s\n", resp1.Content)
	messages = append(messages, litellm.Message{Role: "assistant", Content: resp1.Content})

	// Second turn
	messages = append(messages, litellm.Message{Role: "user", Content: "I want to scrape product information from an e-commerce site. What libraries should I use?"})
	resp2, err := client.Complete(context.Background(), &litellm.Request{
		Model:       "qwen3-coder-plus",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(300),
		Temperature: litellm.Float64Ptr(0.7),
	})
	if err != nil {
		log.Fatalf("Multi-turn conversation failed: %v", err)
	}

	fmt.Printf("Assistant: %s\n", resp2.Content)

	// Advanced parameters
	fmt.Println("\n--- Advanced Parameters ---")
	advancedResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{Role: "user", Content: "Generate a creative variable name for a machine learning model that predicts stock prices"},
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

	// Test different Qwen models
	fmt.Println("\n--- Testing Different Qwen Models ---")
	models := []string{
		"qwen3-coder-plus",
		"qwen3-coder-flash",
		"qwen3-coder-plus-2025-07-22",
	}

	for _, model := range models {
		fmt.Printf("\nTesting model: %s\n", model)
		modelResp, err := client.Complete(context.Background(), &litellm.Request{
			Model: model,
			Messages: []litellm.Message{
				{Role: "user", Content: "What is your model name and what are you good at?"},
			},
			MaxTokens:   litellm.IntPtr(150),
			Temperature: litellm.Float64Ptr(0.5),
		})
		if err != nil {
			fmt.Printf("Model %s failed: %v\n", model, err)
			continue
		}

		fmt.Printf("Response: %s\n", modelResp.Content)
		fmt.Printf("Tokens used: %d\n", modelResp.Usage.TotalTokens)
	}

	// Test reasoning mode (Qwen3 thinking)
	fmt.Println("\n--- Testing Reasoning Mode ---")
	reasoningResp, err := client.Complete(context.Background(), &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a Python function to implement quicksort algorithm. Explain your approach step by step."},
		},
		MaxTokens:   litellm.IntPtr(1000),
		Temperature: litellm.Float64Ptr(0.1),
		Extra: map[string]interface{}{
			"enable_thinking": true, // Enable Qwen3 reasoning mode
		},
	})

	if err != nil {
		fmt.Printf("Reasoning request failed: %v\n", err)
	} else {
		fmt.Printf("Model: %s\n", reasoningResp.Model)
		fmt.Printf("Final Answer: %s\n", reasoningResp.Content)
		if reasoningResp.Reasoning != nil {
			fmt.Printf("Reasoning Process: %s\n", reasoningResp.Reasoning.Content)
			fmt.Printf("Reasoning Summary: %s\n", reasoningResp.Reasoning.Summary)
			fmt.Printf("Reasoning Tokens: %d\n", reasoningResp.Reasoning.TokensUsed)
		}
		fmt.Printf("Tokens: %d\n", reasoningResp.Usage.TotalTokens)
	}
}
