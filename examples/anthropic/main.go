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
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	client := litellm.New(litellm.WithAnthropic(apiKey))

	fmt.Println("Anthropic Claude Examples - From Simple to Complex")
	fmt.Println("================================================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example")
	fmt.Println("---------------------")
	basicChat(client)

	// Example 2: Streaming Chat
	fmt.Println("\n2. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)

	// Example 3: Function/Tool Calling
	fmt.Println("\n3. Function/Tool Calling Example")
	fmt.Println("--------------------------------")
	functionCalling(client)

	// Example 4: JSON Schema Response Format
	fmt.Println("\n4. JSON Schema Response Format Example")
	fmt.Println("--------------------------------------")
	jsonSchemaExample(client)

	// Example 5: Prompt Caching
	fmt.Println("\n5. Prompt Caching Example")
	fmt.Println("-------------------------")
	promptCachingExample(client)
}

// Example 1: Basic Chat
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "claude-haiku-3.5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What are the main benefits of renewable energy?",
			},
		},
		MaxTokens:   litellm.IntPtr(200),
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

// Example 5: Prompt Caching
func promptCachingExample(client *litellm.Client) {
	ctx := context.Background()

	// Large system prompt that we want to cache
	systemPrompt := `You are an expert code reviewer with deep knowledge of Go, Python, JavaScript, and software architecture.
Your task is to analyze code for:
1. Code quality and best practices
2. Performance optimizations
3. Security vulnerabilities
4. Maintainability issues
5. Documentation completeness

Please provide detailed, actionable feedback with specific examples and suggestions for improvement.
Always explain the reasoning behind your recommendations and consider the broader context of the codebase.`

	// First request - creates cache
	fmt.Println("First request (creates cache):")
	request1 := &litellm.Request{
		Model: "claude-4-sonnet",
		Messages: []litellm.Message{
			{
				Role:         "system",
				Content:      systemPrompt,
				CacheControl: litellm.NewEphemeralCache(), // Cache this large system prompt
			},
			{
				Role:    "user",
				Content: "Review this Go function: func add(a, b int) int { return a + b }",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.3),
	}

	response1, err := client.Chat(ctx, request1)
	if err != nil {
		log.Printf("First request failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response1.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response1.Usage.PromptTokens, response1.Usage.CompletionTokens, response1.Usage.TotalTokens)
	if response1.Usage.CacheCreationInputTokens > 0 {
		fmt.Printf("Cache: %d tokens written to cache\n", response1.Usage.CacheCreationInputTokens)
	}

	// Second request - uses cache
	fmt.Println("\nSecond request (uses cache):")
	request2 := &litellm.Request{
		Model: "claude-4-sonnet",
		Messages: []litellm.Message{
			{
				Role:         "system",
				Content:      systemPrompt, // Same content
				CacheControl: litellm.NewEphemeralCache(),
			},
			{
				Role:    "user",
				Content: "Review this Python function: def multiply(x, y): return x * y",
			},
		},
		MaxTokens:   litellm.IntPtr(500),
		Temperature: litellm.Float64Ptr(0.3),
	}

	response2, err := client.Chat(ctx, request2)
	if err != nil {
		log.Printf("Second request failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response2.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response2.Usage.PromptTokens, response2.Usage.CompletionTokens, response2.Usage.TotalTokens)
	if response2.Usage.CacheReadInputTokens > 0 {
		fmt.Printf("Cache: %d tokens read from cache (cost savings!)\n", response2.Usage.CacheReadInputTokens)
	}
}

// Example 2: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "claude-haiku-3.5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Write a short poem about artificial intelligence",
			},
		},
		MaxTokens:   litellm.IntPtr(150),
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

// Example 3: Function/Tool Calling
func functionCalling(client *litellm.Client) {
	calculatorFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "calculate",
			Description: "Perform basic arithmetic operations",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"operation": map[string]interface{}{
						"type":        "string",
						"description": "The operation to perform",
						"enum":        []string{"add", "subtract", "multiply", "divide"},
					},
					"a": map[string]interface{}{
						"type":        "number",
						"description": "First number",
					},
					"b": map[string]interface{}{
						"type":        "number",
						"description": "Second number",
					},
				},
				"required": []string{"operation", "a", "b"},
			},
		},
	}

	request := &litellm.Request{
		Model: "claude-sonnet-4",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What is 15 multiplied by 23?",
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
			fmt.Printf("Tool Call: %s with arguments: %s\n",
				toolCall.Function.Name, toolCall.Function.Arguments)
		}
	}
}

// Example 4: JSON Schema Response Format
func jsonSchemaExample(client *litellm.Client) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"recommendations": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"title": map[string]interface{}{
							"type": "string",
						},
						"author": map[string]interface{}{
							"type": "string",
						},
						"genre": map[string]interface{}{
							"type": "string",
						},
						"summary": map[string]interface{}{
							"type": "string",
						},
					},
					"required": []string{"title", "author", "genre", "summary"},
				},
			},
		},
		"required": []string{"recommendations"},
	}

	request := &litellm.Request{
		Model: "claude-sonnet-4",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Recommend 3 science fiction books from different decades",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &litellm.JSONSchema{
				Name:        "book_recommendations",
				Description: "A list of book recommendations",
				Schema:      schema,
				Strict:      litellm.BoolPtr(true),
			},
		},
		MaxTokens: litellm.IntPtr(500),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("JSON schema failed: %v", err)
		return
	}

	fmt.Printf("Structured JSON Response: %s\n", response.Content)

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(response.Content), &result); err == nil {
		prettyJSON, _ := json.MarshalIndent(result, "", "  ")
		fmt.Printf("Formatted JSON:\n%s\n", string(prettyJSON))
	}
}
