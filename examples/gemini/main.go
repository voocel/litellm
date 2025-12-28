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
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("gemini", litellm.ProviderConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("Google Gemini Examples - From Simple to Complex")
	fmt.Println("==============================================")

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
}

// Example 1: Basic Chat
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "who are you",
			},
		},
		MaxTokens:   litellm.IntPtr(10000),
		Temperature: litellm.Float64Ptr(0.7),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Basic chat failed: %v", err)
		return
	}

	// Display reasoning content if available
	if response.Reasoning != nil && response.Reasoning.Content != "" {
		fmt.Printf("Reasoning Content: %s\n", response.Reasoning.Content)
		fmt.Printf("Reasoning Tokens: %d\n", response.Reasoning.TokensUsed)
		fmt.Println("---")
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion + %d reasoning = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.ReasoningTokens, response.Usage.TotalTokens)
}

// Example 2: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "你是谁",
			},
		},
		MaxTokens:   litellm.IntPtr(2000),
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
	printPrefix := func(label string, printed *bool) {
		if *printed {
			return
		}
		fmt.Print("\n")
		fmt.Print(label)
		*printed = true
	}

	thinkingPrinted := false
	outputPrinted := false

	_, err = litellm.CollectStreamWithCallbacks(stream, litellm.StreamCallbacks{
		OnContent: func(text string) {
			if text != "" {
				printPrefix("[output]: ", &outputPrinted)
				fmt.Print(text)
			}
		},
		OnReasoning: func(r *litellm.ReasoningChunk) {
			if r.Content != "" {
				printPrefix("[think]: ", &thinkingPrinted)
				fmt.Print(r.Content)
			}
		},
	})
	if err != nil {
		log.Printf("Stream error: %v", err)
	}
	fmt.Println()
}

// Example 3: Function/Tool Calling
func functionCalling(client *litellm.Client) {
	mathFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "calculate_area",
			Description: "Calculate the area of different shapes",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"shape": map[string]interface{}{
						"type":        "string",
						"description": "The shape to calculate area for",
						"enum":        []string{"circle", "rectangle", "triangle"},
					},
					"dimensions": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"radius": map[string]interface{}{
								"type":        "number",
								"description": "Radius for circle",
							},
							"width": map[string]interface{}{
								"type":        "number",
								"description": "Width for rectangle",
							},
							"height": map[string]interface{}{
								"type":        "number",
								"description": "Height for rectangle or triangle",
							},
							"base": map[string]interface{}{
								"type":        "number",
								"description": "Base for triangle",
							},
						},
					},
				},
				"required": []string{"shape", "dimensions"},
			},
		},
	}

	request := &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Calculate the area of a circle with radius 5",
			},
		},
		Tools:     []litellm.Tool{mathFunction},
		MaxTokens: litellm.IntPtr(2000),
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
			"movie": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"title": map[string]interface{}{
						"type": "string",
					},
					"genre": map[string]interface{}{
						"type": "string",
					},
					"year": map[string]interface{}{
						"type": "integer",
					},
					"rating": map[string]interface{}{
						"type": "number",
					},
					"description": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"title", "genre", "year", "description"},
			},
		},
		"required": []string{"movie"},
	}

	request := &litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Recommend a classic science fiction movie",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &litellm.JSONSchema{
				Name:        "movie_recommendation",
				Description: "A movie recommendation with details",
				Schema:      schema,
				Strict:      litellm.BoolPtr(true),
			},
		},
		MaxTokens: litellm.IntPtr(2000),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("JSON schema failed: %v", err)
		return
	}

	fmt.Printf("Structured JSON Response: %s\n", response.Content)

	// Parse and pretty print JSON
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(response.Content), &result); err == nil {
		prettyJSON, _ := json.MarshalIndent(result, "", "  ")
		fmt.Printf("Formatted JSON:\n%s\n", string(prettyJSON))
	}
}
