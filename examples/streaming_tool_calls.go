package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/voocel/litellm"
)

func main() {
	// Initialize client with your API key
	client := litellm.New(litellm.WithOpenAI("your-openai-api-key"))

	// Define tools
	tools := []litellm.Tool{
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "get_weather",
				Description: "Get weather information for a city",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type":        "string",
							"description": "The city name",
						},
						"unit": map[string]interface{}{
							"type":        "string",
							"description": "Temperature unit (celsius or fahrenheit)",
							"enum":        []string{"celsius", "fahrenheit"},
						},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	// Start streaming with tool calls
	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "What's the weather like in Tokyo and New York? Use celsius."},
		},
		Tools:      tools,
		ToolChoice: "auto",
	})

	if err != nil {
		log.Fatalf("Failed to start stream: %v", err)
	}
	defer stream.Close()

	// Track tool calls
	toolCalls := make(map[string]*ToolCallBuilder)

	fmt.Println("Assistant:")
	for {
		chunk, err := stream.Read()
		if err != nil {
			log.Fatalf("Stream read failed: %v", err)
		}

		if chunk.Done {
			fmt.Println("\nStream completed")
			break
		}

		switch chunk.Type {
		case litellm.ChunkTypeContent:
			// Regular content
			fmt.Print(chunk.Content)

		case litellm.ChunkTypeToolCallDelta:
			// Tool call incremental data
			if chunk.ToolCallDelta != nil {
				delta := chunk.ToolCallDelta

				// Get or create tool call builder
				if _, exists := toolCalls[delta.ID]; !exists && delta.ID != "" {
					toolCalls[delta.ID] = &ToolCallBuilder{
						ID:   delta.ID,
						Type: delta.Type,
						Name: delta.FunctionName,
					}
					fmt.Printf("\nTool call started: %s", delta.FunctionName)
				}

				// Accumulate arguments
				if delta.ArgumentsDelta != "" && delta.ID != "" {
					if builder, exists := toolCalls[delta.ID]; exists {
						builder.Arguments.WriteString(delta.ArgumentsDelta)
						fmt.Printf(".")
					}
				}
			}

		case litellm.ChunkTypeToolCall:
			// Complete tool call (for providers like Gemini)
			if len(chunk.ToolCalls) > 0 {
				for _, toolCall := range chunk.ToolCalls {
					fmt.Printf("\nComplete tool call: %s(%s)",
						toolCall.Function.Name, toolCall.Function.Arguments)
				}
			}

		case litellm.ChunkTypeReasoning:
			// Reasoning content (for o-series models)
			if chunk.Reasoning != nil {
				fmt.Printf("\nReasoning: %s", chunk.Reasoning.Summary)
			}
		}
	}

	// Process completed tool calls
	fmt.Println("\n\nTool Calls Summary:")
	for id, builder := range toolCalls {
		fmt.Printf("ID: %s\n", id)
		fmt.Printf("Function: %s\n", builder.Name)
		fmt.Printf("Arguments: %s\n", builder.Arguments.String())
		fmt.Println("---")
	}
}

// ToolCallBuilder helps accumulate tool call data
type ToolCallBuilder struct {
	ID        string
	Type      string
	Name      string
	Arguments strings.Builder
}
