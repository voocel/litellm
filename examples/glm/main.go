package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== GLM-4.5 Provider Examples ===")

	// Check if API key is available
	apiKey := os.Getenv("GLM_API_KEY")
	if apiKey == "" {
		log.Fatal("GLM_API_KEY environment variable is required")
	}

	// Method 1: Auto-discovery from environment variables
	// export GLM_API_KEY="your-api-key"
	client := litellm.New()

	// Method 2: Manual configuration
	// client := litellm.New(litellm.WithGLM("your-api-key"))

	ctx := context.Background()

	// Example 1: Basic Chat
	fmt.Println("\n--- Example 1: Basic Chat ---")
	basicChatExample(ctx, client)

	// Example 2: Code Generation
	fmt.Println("\n--- Example 2: Code Generation ---")
	codeGenerationExample(ctx, client)

	// Example 3: Streaming Response
	fmt.Println("\n--- Example 3: Streaming Response ---")
	streamingExample(ctx, client)

	// Example 4: Function Calling
	fmt.Println("\n--- Example 4: Function Calling ---")
	functionCallingExample(ctx, client)

	// Example 5: Multi-turn Conversation
	fmt.Println("\n--- Example 5: Multi-turn Conversation ---")
	multiTurnExample(ctx, client)

	// Example 6: Thinking Mode (GLM-4.5 Reasoning)
	fmt.Println("\n--- Example 6: Thinking Mode ---")
	thinkingModeExample(ctx, client)

	// Example 7: Different Models
	fmt.Println("\n--- Example 7: Different Models ---")
	differentModelsExample(ctx, client)
}

func basicChatExample(ctx context.Context, client *litellm.Client) {
	response, err := client.Complete(ctx, &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{Role: "user", Content: "Hello! Please introduce the features of the GLM-4.5 model."},
		},
		MaxTokens:   litellm.IntPtr(1000),
		Temperature: litellm.Float64Ptr(0.7),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	fmt.Printf("Model: %s\n", response.Model)
	fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)
}

func codeGenerationExample(ctx context.Context, client *litellm.Client) {
	response, err := client.Complete(ctx, &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{Role: "system", Content: "You are a professional Go developer."},
			{Role: "user", Content: "Write a Go function to calculate the nth Fibonacci number using dynamic programming optimization."},
		},
		MaxTokens:   litellm.IntPtr(1500),
		Temperature: litellm.Float64Ptr(0.3),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Generated Code:\n%s\n", response.Content)
}

func streamingExample(ctx context.Context, client *litellm.Client) {
	stream, err := client.Stream(ctx, &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a poem about artificial intelligence with a rhythmic feel."},
		},
		MaxTokens:   litellm.IntPtr(800),
		Temperature: litellm.Float64Ptr(0.8),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}
	defer stream.Close()

	fmt.Print("Streaming response: ")
	for {
		chunk, err := stream.Read()
		if err != nil {
			break
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()
}

func functionCallingExample(ctx context.Context, client *litellm.Client) {
	tools := []litellm.Tool{
		{
			Type: "function",
			Function: litellm.FunctionSchema{
				Name:        "get_weather",
				Description: "Get weather information for a specified city",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
						"unit": map[string]interface{}{
							"type":        "string",
							"enum":        []string{"celsius", "fahrenheit"},
							"description": "Temperature unit",
						},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	response, err := client.Complete(ctx, &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{Role: "user", Content: "What's the weather like in Beijing today?"},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   litellm.IntPtr(1000),
		Temperature: litellm.Float64Ptr(0.7),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if len(response.ToolCalls) > 0 {
		for _, toolCall := range response.ToolCalls {
			fmt.Printf("Tool Call: %s(%s)\n", toolCall.Function.Name, toolCall.Function.Arguments)
		}
	}
}

func multiTurnExample(ctx context.Context, client *litellm.Client) {
	messages := []litellm.Message{
		{Role: "system", Content: "You are a helpful AI assistant specialized in answering technical questions."},
		{Role: "user", Content: "What is machine learning?"},
	}

	// First turn
	response1, err := client.Complete(ctx, &litellm.Request{
		Model:       "glm-4.5",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(800),
		Temperature: litellm.Float64Ptr(0.7),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("AI: %s\n", response1.Content)

	// Add AI response to conversation
	messages = append(messages, litellm.Message{
		Role:    "assistant",
		Content: response1.Content,
	})

	// Second turn
	messages = append(messages, litellm.Message{
		Role:    "user",
		Content: "Can you give me a specific example of a machine learning algorithm?",
	})

	response2, err := client.Complete(ctx, &litellm.Request{
		Model:       "glm-4.5",
		Messages:    messages,
		MaxTokens:   litellm.IntPtr(800),
		Temperature: litellm.Float64Ptr(0.7),
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("AI: %s\n", response2.Content)
}

func thinkingModeExample(ctx context.Context, client *litellm.Client) {
	response, err := client.Complete(ctx, &litellm.Request{
		Model: "glm-4.5",
		Messages: []litellm.Message{
			{Role: "user", Content: "Design an efficient algorithm to solve the Traveling Salesman Problem (TSP) and analyze its time complexity."},
		},
		MaxTokens:   litellm.IntPtr(2000),
		Temperature: litellm.Float64Ptr(0.3),
		Extra: map[string]interface{}{
			"thinking": map[string]string{
				"type": "enabled", // Enable thinking mode
			},
		},
	})

	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	fmt.Printf("Response: %s\n", response.Content)
	if response.Reasoning != nil {
		fmt.Printf("Reasoning Process: %s\n", response.Reasoning.Content)
		fmt.Printf("Reasoning Summary: %s\n", response.Reasoning.Summary)
		fmt.Printf("Reasoning Tokens: %d\n", response.Reasoning.TokensUsed)
	}
}

func differentModelsExample(ctx context.Context, client *litellm.Client) {
	models := []string{"glm-4.5", "glm-4.5-air", "glm-4.5-flash", "glm-4"}

	for _, model := range models {
		fmt.Printf("\n--- Testing %s ---\n", model)
		response, err := client.Complete(ctx, &litellm.Request{
			Model: model, // Fixed: remove "glm/" prefix
			Messages: []litellm.Message{
				{Role: "user", Content: "Introduce yourself in one sentence."},
			},
			MaxTokens:   litellm.IntPtr(200),
			Temperature: litellm.Float64Ptr(0.7),
		})

		if err != nil {
			log.Printf("Error with %s: %v", model, err)
			continue
		}

		fmt.Printf("%s: %s\n", model, response.Content)
		fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)
	}
}
