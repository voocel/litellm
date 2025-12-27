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
	apiKey := os.Getenv("GLM_API_KEY")
	if apiKey == "" {
		log.Fatal("GLM_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("glm", litellm.ProviderConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("GLM (ZhiPu) Examples - From Basic to Advanced")
	fmt.Println("=============================================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example (GLM-4)")
	fmt.Println("-----------------------------")
	basicChat(client)

	// Example 2: Thinking Mode
	fmt.Println("\n2. Thinking Mode Example")
	fmt.Println("------------------------")
	thinkingMode(client)

	// Example 3: Function Calling
	fmt.Println("\n3. Function Calling Example")
	fmt.Println("---------------------------")
	functionCalling(client)

	// Example 4: Streaming Chat
	fmt.Println("\n4. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)

	// Example 5: JSON Response Format
	fmt.Println("\n5. JSON Response Format Example")
	fmt.Println("-------------------------------")
	jsonResponseFormat(client)
}

// Example 1: Basic Chat with GLM-4
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "you are a helpful AI assistant",
			},
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

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 3: Thinking Mode (GLM special feature)
func thinkingMode(client *litellm.Client) {
	request := &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "There was a farmer who had 17 sheep. All but 9 died. How many sheep are left? Please analyze this question in detail.",
			},
		},
		MaxTokens:   litellm.IntPtr(10000),
		Temperature: litellm.Float64Ptr(0.7),
		Extra: map[string]interface{}{
			"thinking": map[string]interface{}{
				"type": "enabled", // Enable GLM thinking mode
			},
		},
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Thinking mode failed: %v", err)
		return
	}

	// Display thinking process if available
	if response.Reasoning != nil && response.Reasoning.Content != "" {
		fmt.Printf("Thinking Process:\n%s\n", response.Reasoning.Content)
		fmt.Println("---")
	}

	fmt.Printf("Final Answer: %s\n", response.Content)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 4: Function Calling
func functionCalling(client *litellm.Client) {
	// Define a calculator function
	calculatorFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "calculator",
			Description: "Execute mathematical calculations",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "Mathematical expression to evaluate, e.g., '2+3*4'",
					},
				},
				"required": []string{"expression"},
			},
		},
	}

	request := &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "(15 + 25) * 3 - 20 = ?",
			},
		},
		Tools:     []litellm.Tool{calculatorFunction},
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
			fmt.Printf("Tool Call: %s with expression: %s\n",
				toolCall.Function.Name, toolCall.Function.Arguments)
		}
	}
}

// Example 5: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "glm-4.5-air",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Please write a short prose piece about the arrival of spring, around 200 words.\n",
			},
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.8),
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
			fmt.Printf("\n[Thinking: %s]\n", chunk.Reasoning.Content)
		}
	}
	fmt.Println()
}

// Example 6: JSON Response Format
func jsonResponseFormat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "You are a data analysis expert and need to return structured data in JSON format.\n",
			},
			{
				Role:    "user",
				Content: "Please analyze the main characteristics of China's four first-tier cities (Beijing, Shanghai, Guangzhou, Shenzhen), including information such as population, GDP, and major industries, and return the results in JSON format.",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_object",
		},
		MaxTokens:   litellm.IntPtr(800),
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
