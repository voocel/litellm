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
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
		APIKey:  apiKey,
		BaseURL: os.Getenv("OPENAI_BASE_URL"),
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("OpenAI Examples - From Simple to Complex")
	fmt.Println("=====================================")

	//Example 1: Basic Chat
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

	//Example 5: Responses API
	fmt.Println("\n5. Responses API Example")
	fmt.Println("------------------------")
	responsesAPI(client)
}

// Example 5: Responses API (for detailed reasoning)
func responsesAPI(client *litellm.Client) {
	request := &litellm.OpenAIResponsesRequest{
		Model: "o3-mini",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Solve this step by step: A train travels at 80 km/h for 2.5 hours, then at 120 km/h for 1.5 hours. What is the average speed?",
			},
		},
		ReasoningEffort:  "high",
		ReasoningSummary: "auto",
		MaxOutputTokens:  litellm.IntPtr(800),
	}

	ctx := context.Background()
	response, err := client.Responses(ctx, request)
	if err != nil {
		log.Printf("Responses API failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if response.Reasoning != nil {
		fmt.Printf("Reasoning Summary: %s\n", response.Reasoning.Summary)
	}
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 1: Basic Chat
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Who are you?",
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

// Example 2: Streaming Chat with Reasoning Models (gpt-5, o1, o3, o4)
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gpt-5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "who are you?",
			},
		},
		MaxTokens: litellm.IntPtr(10000),
	}

	stream, err := client.Stream(context.Background(), request)
	if err != nil {
		log.Printf("Streaming failed: %v", err)
		return
	}
	defer stream.Close()

	fmt.Print("\nStreaming response: ")
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

	response, err := litellm.CollectStreamWithCallbacks(stream, litellm.StreamCallbacks{
		OnContent: func(text string) {
			if text != "" {
				printPrefix("output: ", &outputPrinted)
				fmt.Print(text)
			}
		},
		OnReasoning: func(r *litellm.ReasoningChunk) {
			if r.Content != "" {
				printPrefix("think: ", &thinkingPrinted)
				fmt.Print(r.Content)
			}
			if r.Content == "" && r.Summary != "" {
				printPrefix("think: ", &thinkingPrinted)
				fmt.Print(r.Summary)
			}
		},
	})
	if err != nil {
		log.Printf("\nStream error: %v", err)
		return
	}
	fmt.Println()

	// Print token usage statistics
	if response.Usage.TotalTokens > 0 {
		fmt.Println("\nToken Usage:")
		fmt.Printf("  Prompt Tokens:     %d\n", response.Usage.PromptTokens)
		fmt.Printf("  Completion Tokens: %d\n", response.Usage.CompletionTokens)
		if response.Usage.ReasoningTokens > 0 {
			fmt.Printf("  Reasoning Tokens:  %d (hidden internal reasoning)\n", response.Usage.ReasoningTokens)
		}
		fmt.Printf("  Total Tokens:      %d\n", response.Usage.TotalTokens)
	}
}

// Example 3: Function/Tool Calling
func functionCalling(client *litellm.Client) {
	weatherFunction := litellm.Tool{
		Type: "function",
		Function: litellm.FunctionDef{
			Name:        "get_current_weather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
					"unit": map[string]interface{}{
						"type": "string",
						"enum": []string{"celsius", "fahrenheit"},
					},
				},
				"required": []string{"location"},
			},
		},
	}

	request := &litellm.Request{
		Model: "gpt-4o",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "What's the weather like in Tokyo?",
			},
		},
		Tools:      []litellm.Tool{weatherFunction},
		ToolChoice: "auto",
		MaxTokens:  litellm.IntPtr(150),
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
	// Define JSON schema for structured response
	// Note: additionalProperties: false will be automatically added by LiteLLM for OpenAI
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"person": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type": "string",
					},
					"age": map[string]interface{}{
						"type": "integer",
					},
					"occupation": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"name", "age", "occupation"},
			},
		},
		"required": []string{"person"},
	}

	request := &litellm.Request{
		Model: "gpt-5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Create a person profile for a 30-year-old software engineer named Alice",
			},
		},
		ResponseFormat: &litellm.ResponseFormat{
			Type: "json_schema",
			JSONSchema: &litellm.JSONSchema{
				Name:        "person_profile",
				Description: "A person's profile information",
				Schema:      schema,
				Strict:      litellm.BoolPtr(true),
			},
		},
		MaxTokens: litellm.IntPtr(200),
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
