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
	apiKey := os.Getenv("MINIMAX_API_KEY")
	if apiKey == "" {
		log.Fatal("MINIMAX_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("minimax", litellm.ProviderConfig{
		APIKey: apiKey,
		// China-region users can set BaseURL: "https://api.minimaxi.com/v1".
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("MiniMax Examples - MiniMax M3")
	fmt.Println("==============================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat Example (MiniMax-M3)")
	fmt.Println("----------------------------------")
	basicChat(client)

	// Example 2: Chat with Thinking Enabled
	fmt.Println("\n2. Chat with Thinking Enabled (MiniMax-M3)")
	fmt.Println("------------------------------------------")
	chatWithThinking(client)

	// Example 3: Function Calling
	fmt.Println("\n3. Function Calling Example")
	fmt.Println("---------------------------")
	functionCalling(client)

	// Example 4: Streaming Chat
	fmt.Println("\n4. Streaming Chat Example")
	fmt.Println("-------------------------")
	streamingChat(client)
}

// Example 1: Basic Chat with MiniMax-M3
func basicChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "MiniMax-M3",
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

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 2: Chat with Thinking (reasoning) Enabled
//
// MiniMax-M3 supports deep thinking via the `thinking` parameter.
// When enabled (type: "adaptive"), the model decides whether to
// engage in deep reasoning before answering. The reasoning content
// is returned in the Response.ReasoningContent field.
func chatWithThinking(client *litellm.Client) {
	request := &litellm.Request{
		Model: "MiniMax-M3",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Which is bigger, 9.11 or 9.9? Please think through this step by step.",
			},
		},
		MaxTokens:   litellm.IntPtr(1000),
		Temperature: litellm.Float64Ptr(0.7),
		Thinking: &litellm.ThinkingConfig{
			Type: "enabled", // MiniMax maps "enabled" → "adaptive"
		},
	}

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Chat with thinking failed: %v", err)
		return
	}

	// Display reasoning if available
	if response.ReasoningContent != "" {
		fmt.Printf("Reasoning Process:\n%s\n", response.ReasoningContent)
		fmt.Println("---")
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Usage: %d prompt + %d completion = %d total tokens\n",
		response.Usage.PromptTokens, response.Usage.CompletionTokens, response.Usage.TotalTokens)
}

// Example 3: Function Calling
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
		Model: "MiniMax-M3",
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
			// Pretty print the arguments JSON
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err == nil {
				prettyArgs, _ := json.MarshalIndent(args, "", "  ")
				fmt.Printf("Tool Call: %s\n", toolCall.Function.Name)
				fmt.Printf("Arguments: %s\n", string(prettyArgs))
			} else {
				fmt.Printf("Tool Call: %s with arguments: %s\n",
					toolCall.Function.Name, toolCall.Function.Arguments)
			}
		}
	}
	fmt.Printf("Usage: %d tokens\n", response.Usage.TotalTokens)
}

// Example 4: Streaming Chat
func streamingChat(client *litellm.Client) {
	request := &litellm.Request{
		Model: "MiniMax-M3",
		Messages: []litellm.Message{
			{
				Role:    "user",
				Content: "Write a short poem about artificial intelligence.",
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
		OnReasoning: func(content string) {
			if content != "" {
				printPrefix("[think]: ", &thinkingPrinted)
				fmt.Print(content)
			}
		},
	})
	if err != nil {
		log.Printf("Stream error: %v", err)
	}
	fmt.Println()
}
