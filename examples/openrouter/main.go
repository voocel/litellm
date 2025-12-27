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
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("openrouter", litellm.ProviderConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("OpenRouter Examples - Access Multiple AI Models")
	fmt.Println("==============================================")

	// Example 1: OpenAI via OpenRouter
	fmt.Println("\n1. OpenAI via OpenRouter")
	fmt.Println("-----------------------")
	testOpenAI(client)

	// Example 2: Claude 3.5 Sonnet via OpenRouter
	fmt.Println("\n2. Claude 3.5 Sonnet via OpenRouter")
	fmt.Println("-----------------------------------")
	testClaude(client)

	// Example 3: Llama 3.3 via OpenRouter
	fmt.Println("\n3. Llama 3.3 70B via OpenRouter")
	fmt.Println("-------------------------------")
	testLlama(client)

	// Example 4: DeepSeek via OpenRouter
	fmt.Println("\n4. DeepSeek Chat via OpenRouter")
	fmt.Println("-------------------------------")
	testDeepSeek(client)

	// Example 5: Function Calling with Different Models
	fmt.Println("\n5. Function Calling Example")
	fmt.Println("---------------------------")
	functionCalling(client)

	// Example 6: Streaming Chat
	fmt.Println("\n6. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)

	// Example 7: JSON Schema Response
	fmt.Println("\n7. JSON Schema Response Example")
	fmt.Println("-------------------------------")
	jsonSchemaResponse(client)
}

// Example 1: OpenAI via OpenRouter
func testOpenAI(client *litellm.Client) {
	request := &litellm.Request{
		Model: "openai/gpt-5-nano",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What model are you?",
			},
		},
		MaxTokens:   litellm.IntPtr(2000),
		Temperature: litellm.Float64Ptr(0.7),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("GPT-4o via OpenRouter failed: %v", err)
		return
	}

	if response.Reasoning != nil && response.Reasoning.Content != "" {
		fmt.Printf("Thinking Process:\n%s\n", response.Reasoning.Content)
		fmt.Println("---")
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model Used: %s\n", response.Model)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 2: Claude 3.5 haiku via OpenRouter
func testClaude(client *litellm.Client) {
	request := &litellm.Request{
		Model: "anthropic/claude-3.5-haiku",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What model are you?",
			},
		},
		MaxTokens:   litellm.IntPtr(300),
		Temperature: litellm.Float64Ptr(0.8),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Claude via OpenRouter failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model Used: %s\n", response.Model)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 3: Llama 3.3 via OpenRouter
func testLlama(client *litellm.Client) {
	request := &litellm.Request{
		Model: "meta-llama/llama-3.3-70b-instruct",
		Messages: []litellm.Message{
			{
				Role:    "system",
				Content: "You are a helpful coding assistant.",
			},
			{
				Role:    "user",
				Content: "Write a Python function to calculate the Fibonacci sequence using dynamic programming.",
			},
		},
		MaxTokens:   litellm.IntPtr(600),
		Temperature: litellm.Float64Ptr(0.4),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Llama via OpenRouter failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model Used: %s\n", response.Model)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 4: DeepSeek via OpenRouter
func testDeepSeek(client *litellm.Client) {
	request := &litellm.Request{
		Model: "deepseek/deepseek-chat",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What are the key advantages of using AI model routing services like OpenRouter?",
			},
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.6),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("DeepSeek via OpenRouter failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model Used: %s\n", response.Model)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 5: Function Calling with Different Models
func functionCalling(client *litellm.Client) {
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
		Model: "openai/gpt-4o-mini", // Use a cost-effective model for function calling
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What's the weather like in Tokyo?",
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
	fmt.Printf("Model Used: %s\n", response.Model)
}

// Example 6: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "google/gemini-2.5-flash", // Use Gemini via OpenRouter for streaming
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Write a short story about a developer who discovers they can access any AI model through a single API.",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
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
			fmt.Printf("\n[Reasoning: %s]\n", chunk.Reasoning.Content)
		}
	}
	fmt.Println()
}

// Example 7: JSON Schema Response
func jsonSchemaResponse(client *litellm.Client) {
	// Define a schema for product information
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"product": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type": "string",
					},
					"category": map[string]interface{}{
						"type": "string",
					},
					"price": map[string]interface{}{
						"type": "number",
					},
					"features": map[string]interface{}{
						"type": "array",
						"items": map[string]interface{}{
							"type": "string",
						},
					},
					"rating": map[string]interface{}{
						"type":    "number",
						"minimum": 0,
						"maximum": 5,
					},
				},
				"required": []string{"name", "category", "price", "features", "rating"},
			},
		},
		"required": []string{"product"},
	}

	request := &litellm.Request{
		Model: "openai/gpt-4o-mini",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Generate information for a high-end wireless headphone product.",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &litellm.JSONSchema{
				Name:        "product_info",
				Description: "Product information with structured data",
				Schema:      schema,
				Strict:      litellm.BoolPtr(true),
			},
		},
		MaxTokens:   litellm.IntPtr(400),
		Temperature: litellm.Float64Ptr(0.5),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("JSON schema response failed: %v", err)
		return
	}

	fmt.Printf("JSON Response: %s\n", response.Content)

	// Parse and pretty print JSON
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(response.Content), &result); err == nil {
		prettyJSON, _ := json.MarshalIndent(result, "", "  ")
		fmt.Printf("Formatted JSON:\n%s\n", string(prettyJSON))
	}

	fmt.Printf("Model Used: %s\n", response.Model)
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}
