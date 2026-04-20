package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	apiKey := os.Getenv("GROK_API_KEY")
	if apiKey == "" {
		log.Fatal("GROK_API_KEY environment variable is required")
	}

	client, err := litellm.NewWithProvider("grok", litellm.ProviderConfig{
		APIKey: apiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("Grok (xAI) Examples")
	fmt.Println("====================")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat")
	fmt.Println("-------------")
	basicChat(client)

	// Example 2: Streaming with Reasoning (grok-4.1-thinking supports reasoning_content)
	fmt.Println("\n2. Streaming with Reasoning")
	fmt.Println("---------------------------")
	streamingChat(client)
}

// Example 1: Basic Chat
func basicChat(client *litellm.Client) {
	request := litellm.NewRequest("grok-4.20-0309-reasoning", "Explain what xAI's Grok is in simple terms.",
		litellm.WithMaxTokens(500),
	)

	ctx := context.Background()
	response, err := client.Chat(ctx, request)
	if err != nil {
		log.Printf("Basic chat failed: %v", err)
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

// Example 2: Streaming with Reasoning
// Only grok-4.1-thinking returns reasoning_content in Chat Completions API.
func streamingChat(client *litellm.Client) {
	request := litellm.NewRequest("grok-3", "Write a short poem about artificial intelligence.",
		litellm.WithMaxTokens(400),
		litellm.WithThinking("high"),
	)

	ctx := context.Background()
	stream, err := client.Stream(ctx, request)
	if err != nil {
		log.Printf("Streaming failed: %v", err)
		return
	}
	defer stream.Close()

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

	resp, err := litellm.CollectStreamWithCallbacks(stream, litellm.StreamCallbacks{
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
		return
	}
	fmt.Println()
	fmt.Printf("Usage: %d prompt + %d completion (%d reasoning) = %d total tokens\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.ReasoningTokens, resp.Usage.TotalTokens)
}
