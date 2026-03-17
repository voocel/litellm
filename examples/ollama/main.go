package main

import (
	"context"
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	// Ollama runs locally — no API key needed.
	// Make sure Ollama is running: ollama serve
	client, err := litellm.NewWithProvider("ollama", litellm.ProviderConfig{})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	fmt.Println("Ollama Examples")
	fmt.Println("===============")

	// Example 1: Basic Chat
	fmt.Println("\n1. Basic Chat")
	fmt.Println("-------------")
	basicChat(client)

	// Example 2: Streaming
	fmt.Println("\n2. Streaming")
	fmt.Println("------------")
	streamingChat(client)

	// Example 3: List Models
	fmt.Println("\n3. List Models")
	fmt.Println("--------------")
	listModels(client)
}

func basicChat(client *litellm.Client) {
	resp, err := client.Chat(context.Background(), &litellm.Request{
		Model: "llama3.2",
		Messages: []litellm.Message{
			{Role: "system", Content: "You are a helpful assistant. Be concise."},
			{Role: "user", Content: "What is Ollama?"},
		},
		MaxTokens: litellm.IntPtr(200),
	})
	if err != nil {
		log.Printf("Chat failed: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", resp.Content)
	fmt.Printf("Usage: %d prompt + %d completion tokens\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
}

func streamingChat(client *litellm.Client) {
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "llama3.2",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a haiku about Go programming."},
		},
	})
	if err != nil {
		log.Printf("Stream failed: %v", err)
		return
	}
	defer stream.Close()

	resp, err := litellm.CollectStreamWithHandler(stream, func(chunk *litellm.StreamChunk) {
		if chunk.Type == litellm.ChunkTypeContent && chunk.Content != "" {
			fmt.Print(chunk.Content)
		}
	})
	if err != nil {
		log.Printf("Stream error: %v", err)
		return
	}
	fmt.Printf("\nUsage: %d prompt + %d completion tokens\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
}

func listModels(client *litellm.Client) {
	models, err := client.ListModels(context.Background())
	if err != nil {
		log.Printf("List models failed: %v", err)
		return
	}

	for _, m := range models {
		fmt.Printf("  - %s (%s)\n", m.ID, m.Description)
	}
}
