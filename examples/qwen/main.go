package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	apiKey := os.Getenv("QWEN_API_KEY")
	if apiKey == "" {
		log.Fatal("QWEN_API_KEY environment variable is required")
	}

	client := litellm.New(litellm.WithQwen(apiKey))

	fmt.Println("Qwen Examples - Alibaba Cloud AI")
	fmt.Println("==========================================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example (Qwen-Turbo)")
	fmt.Println("----------------------------------")
	basicChat(client)

	// Example 2: Function Calling
	fmt.Println("\n2. Function Calling Example")
	fmt.Println("---------------------------")
	functionCalling(client)

	// Example 3: Streaming Chat
	fmt.Println("\n3. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)
}

// Example 1: Basic Chat with Qwen-Turbo
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "who are you",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.7),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Basic chat failed: %v", err)
		return
	}

	// Display reasoning if available
	if response.Reasoning != nil && response.Reasoning.Content != "" {
		fmt.Printf("Reasoning Process:\n%s\n", response.Reasoning.Content)
		fmt.Println("---")
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 2: Function Calling
func functionCalling(client *litellm.Client) {
	weatherFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "get_weather",
			Description: "Get the weather information for a specified city",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"city": map[string]interface{}{
						"type":        "string",
						"description": "City names, for example: Beijing, Shanghai, Guangzhou",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "Temperature Units",
					},
				},
				"required": []string{"city"},
			},
		},
	}

	request := &litellm.Request{
		Model: "qwen-plus",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Please help me check what the weather is like in Beijing today.",
			},
		},
		Tools:     []litellm.Tool{weatherFunction},
		MaxTokens: litellm.IntPtr(300),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Function calling failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if len(response.ToolCalls) > 0 {
		for _, toolCall := range response.ToolCalls {
			fmt.Printf("Tool Call: %s with arguments: %s\n",
				toolCall.Function.Name, toolCall.Function.Arguments)
		}
	}
}

// Example 3: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "qwen3-coder-plus",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "who are you",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.8),
		Stream:      true,
	}

	ctx := context.Background()
	stream, err := client.Stream(ctx, request)
	if err != nil {
		log.Printf("Streaming failed: %v", err)
		return
	}
	defer stream.Close()

	fmt.Print("Streaming response: ")
	for {
		chunk, err := stream.Next()
		if err != nil {
			log.Printf("Stream error: %v", err)
			break
		}

		if chunk.Done {
			break
		}

		if chunk.Type == "content" && chunk.Content != "" {
			fmt.Print(chunk.Content)
		}

		if chunk.Type == "reasoning" && chunk.Reasoning != nil {
			fmt.Printf("\n[reasoning: %s]\n", chunk.Reasoning.Content)
		}
	}
	fmt.Println()
}
