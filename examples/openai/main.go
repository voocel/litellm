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

	client := litellm.New(litellm.WithOpenAI(apiKey, os.Getenv("OPENAI_BASE_URL")))

	fmt.Println("OpenAI Examples - From Simple to Complex")
	fmt.Println("=====================================")

	// Example 1: Basic Chat
	//fmt.Println("\n1. Basic Chat Example")
	//fmt.Println("---------------------")
	//basicChat(client)

	//// Example 2: Streaming Chat
	//fmt.Println("\n2. Streaming Chat Example")
	//fmt.Println("-------------------------")
	//streamingChat(client)
	//
	//// Example 3: Function/Tool Calling
	//fmt.Println("\n3. Function/Tool Calling Example")
	//fmt.Println("--------------------------------")
	//functionCalling(client)
	//
	//// Example 4: JSON Schema Response Format
	fmt.Println("\n4. JSON Schema Response Format Example")
	fmt.Println("--------------------------------------")
	jsonSchemaExample(client)
	//
	//// Example 5: Reasoning Models (o1, o3, o4, gpt-5)
	//// Note: o1 models use built-in reasoning without explicit parameters
	//fmt.Println("\n5. Reasoning Models Example")
	//fmt.Println("---------------------------")
	//reasoningModels(client)
	//
	//// Example 6: Responses API
	//fmt.Println("\n6. Responses API Example")
	//fmt.Println("------------------------")
	//responsesAPI(client)
}

// Example 1: Basic Chat
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gpt-4.1-mini",
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

// Example 2: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Tell me a short story about a robot",
			},
		},
		MaxTokens:   litellm.IntPtr(100),
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
		Model: "gpt-4o",
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

// Example 5: Reasoning Models (o1, o3, o4, gpt-5)
// Note: o3 models work best with Responses API to show thinking process
func reasoningModels(client *litellm.Client) {
	request := &litellm.Request{
		Model: "o3-mini",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Solve this step by step: If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is the average speed for the entire journey?",
			},
		},
		// Use Responses API to get thinking content
		UseResponsesAPI:  true,
		ReasoningEffort:  "medium",
		ReasoningSummary: "detailed",
		MaxTokens:        litellm.IntPtr(500),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Reasoning model failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if response.Reasoning != nil {
		fmt.Printf("Reasoning Summary: %s\n", response.Reasoning.Summary)
		if response.Reasoning.Content != "" {
			fmt.Printf("Reasoning Process: %s\n", response.Reasoning.Content)
		}
		fmt.Printf("Reasoning Tokens Used: %d\n", response.Usage.ReasoningTokens)
	}
}

// Example 6: Responses API
func responsesAPI(client *litellm.Client) {
	request := &litellm.Request{
		Model: "gpt-5",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Explain quantum computing in simple terms, then provide a detailed technical explanation",
			},
		},
		UseResponsesAPI:  true,
		ReasoningEffort:  "high",
		ReasoningSummary: "comprehensive",
		MaxTokens:        litellm.IntPtr(800),
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Responses API failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if response.Reasoning != nil {
		fmt.Printf("Comprehensive Reasoning: %s\n", response.Reasoning.Summary)
		fmt.Printf("Reasoning Tokens: %d\n", response.Usage.ReasoningTokens)
	}
	fmt.Printf("Total Usage: %d tokens\n", response.Usage.TotalTokens)
}
