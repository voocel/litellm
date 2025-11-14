package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	// Get API keys from environment
	openaiKey := os.Getenv("OPENAI_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	
	if openaiKey == "" || anthropicKey == "" {
		log.Fatal("Both OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables are required for this example")
	}

	fmt.Println("Advanced Routing Examples")
	fmt.Println("========================")

	// Example 1: Default Auto Router (intelligent routing)
	fmt.Println("\n1. Default Auto Router Example")
	fmt.Println("------------------------------")
	defaultRouterExample(openaiKey, anthropicKey)

	// Example 2: Exact Router (strict model matching)
	fmt.Println("\n2. Exact Router Example")
	fmt.Println("-----------------------")
	exactRouterExample(openaiKey, anthropicKey)

	// Example 3: Round Robin Router (load balancing)
	fmt.Println("\n3. Round Robin Router Example")
	fmt.Println("-----------------------------")
	roundRobinExample(openaiKey, anthropicKey)

	// Example 4: Custom Router (user-defined logic)
	fmt.Println("\n4. Custom Router Example")
	fmt.Println("------------------------")
	customRouterExample(openaiKey, anthropicKey)
}

// Example 1: Default Auto Router with intelligent model routing
func defaultRouterExample(openaiKey, anthropicKey string) {
	client, err := litellm.New(
		litellm.WithOpenAI(openaiKey),
		litellm.WithAnthropic(anthropicKey),
		// Using default auto router (no WithRouter needed)
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	ctx := context.Background()
	
	// Test intelligent routing - should route to OpenAI
	fmt.Println("Testing 'gpt-4o-mini' (should route to OpenAI):")
	response, err := client.Chat(ctx, &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "Hello from auto router!"},
		},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Provider: %s, Response: %s\n", response.Provider, response.Content)
	}

	// Test fuzzy matching - "gpt4" should match "gpt-4o-mini"
	fmt.Println("Testing 'gpt4' (fuzzy matching):")
	response2, err := client.Chat(ctx, &litellm.Request{
		Model: "gpt4",
		Messages: []litellm.Message{
			{Role: "user", Content: "Hello from fuzzy matching!"},
		},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Provider: %s, Response: %s\n", response2.Provider, response2.Content)
	}
}

// Example 2: Exact Router requires precise model names
func exactRouterExample(openaiKey, anthropicKey string) {
	client, err := litellm.New(
		litellm.WithOpenAI(openaiKey),
		litellm.WithAnthropic(anthropicKey),
		litellm.WithRouter(litellm.NewExactRouter()), // Requires exact matches
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	ctx := context.Background()
	
	// This should work with exact model name
	fmt.Println("Testing exact match 'gpt-4o-mini':")
	response, err := client.Chat(ctx, &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "Hello from exact router!"},
		},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Provider: %s, Response: %s\n", response.Provider, response.Content)
	}

	// This should fail with exact router (no fuzzy matching)
	fmt.Println("Testing fuzzy match 'gpt4' (should fail with exact router):")
	_, err2 := client.Chat(ctx, &litellm.Request{
		Model: "gpt4",
		Messages: []litellm.Message{
			{Role: "user", Content: "This should fail"},
		},
		MaxTokens: litellm.IntPtr(50),
	})
	if err2 != nil {
		fmt.Printf("Expected error: %v\n", err2)
	}
}

// Example 3: Round Robin Router for load balancing
func roundRobinExample(openaiKey, anthropicKey string) {
	client, err := litellm.New(
		litellm.WithOpenAI(openaiKey),
		litellm.WithAnthropic(anthropicKey),
		litellm.WithRouter(litellm.NewRoundRobinRouter()),
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	ctx := context.Background()
	
	// Make multiple requests with same model - should distribute across providers
	for i := 0; i < 4; i++ {
		response, err := client.Chat(ctx, &litellm.Request{
			Model: "gpt-4o-mini", // Both providers should support this conceptually
			Messages: []litellm.Message{
				{Role: "user", Content: fmt.Sprintf("Request %d from round robin", i+1)},
			},
			MaxTokens: litellm.IntPtr(30),
		})
		if err != nil {
			log.Printf("Request %d error: %v", i+1, err)
		} else {
			fmt.Printf("Request %d - Provider: %s\n", i+1, response.Provider)
		}
	}
}

// Example 4: Custom Router with user-defined logic
func customRouterExample(openaiKey, anthropicKey string) {
	// Custom router that prefers Anthropic for creative tasks
	customRouter := litellm.CustomRouterFunc(func(model string, providers []litellm.Provider) (litellm.Provider, error) {
		// If model contains "creative" or "story", prefer Anthropic
		if model == "creative-writer" {
			for _, provider := range providers {
				if provider.Name() == "anthropic" {
					return provider, nil
				}
			}
		}
		
		// For technical tasks, prefer OpenAI
		if model == "code-helper" {
			for _, provider := range providers {
				if provider.Name() == "openai" {
					return provider, nil
				}
			}
		}
		
		// Default to first available provider
		if len(providers) > 0 {
			return providers[0], nil
		}
		
		return nil, litellm.NewError(litellm.ErrorTypeValidation, "no providers available")
	})

	client, err := litellm.New(
		litellm.WithOpenAI(openaiKey),
		litellm.WithAnthropic(anthropicKey),
		litellm.WithRouter(customRouter),
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	ctx := context.Background()

	// Test creative task routing
	fmt.Println("Testing custom router with 'creative-writer':")
	response, err := client.Chat(ctx, &litellm.Request{
		Model: "creative-writer",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a short poem about routing"},
		},
		MaxTokens: litellm.IntPtr(100),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Provider: %s, Response: %s\n", response.Provider, response.Content)
	}

	// Test technical task routing
	fmt.Println("Testing custom router with 'code-helper':")
	response2, err := client.Chat(ctx, &litellm.Request{
		Model: "code-helper",
		Messages: []litellm.Message{
			{Role: "user", Content: "Write a simple Go function"},
		},
		MaxTokens: litellm.IntPtr(100),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Provider: %s, Response: %s\n", response2.Provider, response2.Content)
	}
}