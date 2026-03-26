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
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("OpenAI Examples - From Simple to Complex")
	fmt.Println("=====================================")

	// Example 1: Basic Responses API
	fmt.Println("\n1. Basic Reasoning Example")
	fmt.Println("--------------------------")
	basicReasoning(client)

	// Example 2: Streaming Chat
	fmt.Println("\n2. Streaming Reasoning Example")
	fmt.Println("------------------------------")
	streamingChat(client)

	// Example 3: Chat Completions reasoning probe
	fmt.Println("\n3. Chat Completions Reasoning Probe")
	fmt.Println("-----------------------------------")
	chatCompletionsReasoning(client)

	// Example 3: Function/Tool Calling
	fmt.Println("\n3. Function/Tool Calling Example")
	fmt.Println("--------------------------------")
	functionCalling(client)

	// Example 4: JSON Schema Response Format
	fmt.Println("\n4. JSON Schema Response Format Example")
	fmt.Println("--------------------------------------")
	jsonSchemaExample(client)
}

// Example 1: Basic reasoning via Responses API.
// 这里只打印官方 reasoning 字段，不通过提示词诱导模型把“思考摘要”写进正文。
func basicReasoning(client *litellm.Client) {
	request := &litellm.OpenAIResponsesRequest{
		Model: "gpt-5.4",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "思考一下为什么天空看起来是蓝色的？",
			},
		},
		ReasoningEffort:  "high",
		ReasoningSummary: "detailed",
		MaxOutputTokens:  litellm.IntPtr(2200),
	}

	ctx := context.Background()
	response, err := client.Responses(ctx, request)
	if err != nil {
		log.Printf("Basic reasoning failed: %v", err)
		return
	}

	if response.ReasoningContent != "" {
		fmt.Printf("Reasoning: %s\n", response.ReasoningContent)
	}
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion (%d reasoning) = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.ReasoningTokens, response.Usage.TotalTokens)

	if cost, err := litellm.CalculateCostForResponse(response); err == nil {
		fmt.Printf("Cost: $%.6f (input: $%.6f, output: $%.6f)\n", cost.TotalCost, cost.InputCost, cost.OutputCost)
	} else {
		fmt.Printf("Cost calculation: %v\n", err)
	}
}

// Example 2: Streaming reasoning via Responses API.
// OnReasoning 只消费官方 reasoning 流事件，不依赖正文内容。
func streamingChat(client *litellm.Client) {
	request := &litellm.OpenAIResponsesRequest{
		Model: "gpt-5.4",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "一个人先以每小时 80 公里行驶 2.5 小时，再以每小时 120 公里行驶 1.5 小时。平均速度是多少？",
			},
		},
		ReasoningEffort:  "high",
		MaxOutputTokens:  litellm.IntPtr(2200),
		ReasoningSummary: "auto",
	}

	stream, err := client.ResponsesStream(context.Background(), request)
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
		OnReasoning: func(content string) {
			if content != "" {
				printPrefix("Reasoning: ", &thinkingPrinted)
				fmt.Print(content)
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
			fmt.Printf("  Reasoning Tokens:  %d\n", response.Usage.ReasoningTokens)
		}
		fmt.Printf("  Total Tokens:      %d\n", response.Usage.TotalTokens)
	}
}

// Example 3: Chat Completions reasoning probe.
// 这个例子用于验证 Chat Completions 是否返回可见的 reasoning 字段。
func chatCompletionsReasoning(client *litellm.Client) {
	request := litellm.NewRequest("gpt-5.4", "一个人先以每小时 80 公里行驶 2.5 小时，再以每小时 120 公里行驶 1.5 小时。平均速度是多少？",
		litellm.WithMaxTokens(2200),
		litellm.WithThinking("high"),
	)

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Chat Completions probe failed: %v", err)
		return
	}

	if response.ReasoningContent != "" {
		fmt.Printf("Reasoning: %s\n", response.ReasoningContent)
	}
	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion (%d reasoning) = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.ReasoningTokens, response.Usage.TotalTokens)
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
