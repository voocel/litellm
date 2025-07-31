package main

import (
	"context"
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	fmt.Println("=== Multi-Provider Basic Usage ===")

	// Method 1: Auto-discovery from environment variables
	// export OPENAI_API_KEY="your-key"
	// export ANTHROPIC_API_KEY="your-key"
	// export GEMINI_API_KEY="your-key"
	// export DEEPSEEK_API_KEY="your-key"
	client := litellm.New()

	// Method 2: Manual configuration for multiple providers
	// client := litellm.New(
	//     litellm.WithOpenAI("your-openai-key"),
	//     litellm.WithAnthropic("your-anthropic-key"),
	//     litellm.WithGemini("your-gemini-key"),
	//     litellm.WithDeepSeek("your-deepseek-key"),
	// )

	// Test different models from different providers
	models := []string{
		"gpt-4o-mini",      // OpenAI
		"claude-4-sonnet",   // Anthropic
		"gemini-2.5-flash", // Google
		"deepseek-chat",    // DeepSeek
	}

	question := "What is the capital of France?"

	for _, model := range models {
		fmt.Printf("\n--- Testing %s ---\n", model)

		response, err := client.Complete(context.Background(), &litellm.Request{
			Model: model,
			Messages: []litellm.Message{
				{Role: "user", Content: question},
			},
			MaxTokens:   litellm.IntPtr(50),
			Temperature: litellm.Float64Ptr(0.3),
		})
		if err != nil {
			fmt.Printf("❌ Error with %s: %v\n", model, err)
			continue
		}

		fmt.Printf("Provider: %s\n", response.Provider)
		fmt.Printf("Response: %s\n", response.Content)
		fmt.Printf("Tokens: %d\n", response.Usage.TotalTokens)
	}

	// Quick call example
	fmt.Println("\n=== Quick Call Example ===")
	quickResp, err := litellm.Quick("gpt-4o-mini", "Hello, LiteLLM!")
	if err != nil {
		log.Printf("Quick call failed: %v", err)
	} else {
		fmt.Printf("Quick response: %s\n", quickResp.Content)
	}
}
