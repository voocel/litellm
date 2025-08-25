package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatal("DEEPSEEK_API_KEY environment variable is required")
	}

	client := litellm.New(litellm.WithDeepSeek(apiKey))

	fmt.Println("DeepSeek Examples - From Basic to Advanced")
	fmt.Println("=========================================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example (DeepSeek Chat)")
	fmt.Println("-------------------------------------")
	basicChat(client)

	// Example 2: Reasoning Mode
	fmt.Println("\n2. Reasoning Mode Example (DeepSeek Reasoner)")
	fmt.Println("--------------------------------------------")
	reasoningChat(client)

	// Example 3: Code Generation
	fmt.Println("\n3. Code Generation Example (DeepSeek Coder)")
	fmt.Println("------------------------------------------")
	codeGeneration(client)

	// Example 4: Function Calling
	fmt.Println("\n4. Function Calling Example")
	fmt.Println("---------------------------")
	functionCalling(client)

	// Example 5: Streaming Chat
	fmt.Println("\n5. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)

	// Example 6: JSON Response Format
	fmt.Println("\n6. JSON Response Format Example")
	fmt.Println("-------------------------------")
	jsonResponseFormat(client)
}

// Example 1: Basic Chat with DeepSeek Chat
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "You are a helpful AI assistant.",
			},
			{
				Role:    "user",
				Content: "Explain what DeepSeek is in simple terms.",
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

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 2: Reasoning Mode with DeepSeek Reasoner
func reasoningChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-reasoner",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "A farmer has 17 sheep, and all but 9 die. How many sheep are left? Think through this step by step.",
			},
		},
		MaxTokens:   litellm.IntPtr(2000),
		Temperature: litellm.Float64Ptr(0.3),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Reasoning chat failed: %v", err)
		return
	}

	// Display reasoning if available
	if response.Reasoning != nil && response.Reasoning.Content != "" {
		fmt.Printf("Reasoning Process:\n%s\n", response.Reasoning.Content)
		fmt.Printf("Reasoning Tokens: %d\n", response.Reasoning.TokensUsed)
		fmt.Println("---")
	} else {
		fmt.Println("DEBUG: No reasoning content found in response")
		fmt.Printf("DEBUG: Response object - Reasoning field: %v\n", response.Reasoning)
		fmt.Printf("DEBUG: Usage - ReasoningTokens: %d, CompletionTokens: %d\n",
			response.Usage.ReasoningTokens, response.Usage.CompletionTokens)
	}

	fmt.Printf("Final Answer: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion + %d reasoning = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.ReasoningTokens, response.Usage.TotalTokens)
}

// Example 3: Code Generation with DeepSeek Coder
func codeGeneration(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-coder",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Write a Python function to find the longest common subsequence of two strings. Include comments and examples.",
			},
		},
		MaxTokens:   litellm.IntPtr(800),
		Temperature: litellm.Float64Ptr(0.2),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Code generation failed: %v", err)
		return
	}

	fmt.Printf("Generated Code:\n%s\n", response.Content)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 4: Function Calling
func functionCalling(client *litellm.Client) {
	// Define a weather function
	weatherFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "get_weather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "The unit of temperature",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	request := &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What's the weather like in Beijing?",
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

// Example 5: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Write a short story about artificial intelligence in 3 paragraphs",
			},
		},
		MaxTokens:   litellm.IntPtr(400),
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
	}
	fmt.Println()
}

// Example 6: JSON Response Format
func jsonResponseFormat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant that responds only in valid JSON format.",
			},
			{
				Role:    "user",
				Content: "Generate a product description for a smart phone with name, price, features, and specifications in JSON format.",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_object",
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.5),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("JSON response failed: %v", err)
		return
	}

	fmt.Printf("JSON Response: %s\n", response.Content)

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(response.Content), &result); err == nil {
		prettyJSON, _ := json.MarshalIndent(result, "", "  ")
		fmt.Printf("Formatted JSON:\n%s\n", string(prettyJSON))
	}

	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}
