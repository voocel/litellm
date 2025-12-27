package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")

	if openaiKey == "" || anthropicKey == "" {
		log.Fatal("OPENAI_API_KEY and ANTHROPIC_API_KEY are required")
	}

	openaiClient, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
		APIKey: openaiKey,
	})
	if err != nil {
		log.Fatalf("Failed to create OpenAI client: %v", err)
	}

	anthropicClient, err := litellm.NewWithProvider("anthropic", litellm.ProviderConfig{
		APIKey: anthropicKey,
	})
	if err != nil {
		log.Fatalf("Failed to create Anthropic client: %v", err)
	}

	clients := map[string]*litellm.Client{
		"openai":    openaiClient,
		"anthropic": anthropicClient,
	}

	fmt.Println("Explicit Routing Examples")
	fmt.Println("================")

	// Example 1: explicit provider selection
	explicitProvider(clients)

	// Example 2: manual routing (fully controlled by business rules)
	manualRouting(clients)
}

func explicitProvider(clients map[string]*litellm.Client) {
	ctx := context.Background()
	fmt.Println("\n1) Explicit provider selection")

	resp, err := clients["openai"].Chat(ctx, &litellm.Request{
		Model:     "gpt-4o-mini",
		Messages:  []litellm.Message{{Role: "user", Content: "Hello from OpenAI"}},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("OpenAI request failed: %v", err)
		return
	}
	fmt.Printf("OpenAI -> %s\n", resp.Content)

	resp, err = clients["anthropic"].Chat(ctx, &litellm.Request{
		Model:     "claude-sonnet-4-5-20250929",
		Messages:  []litellm.Message{{Role: "user", Content: "Hello from Anthropic"}},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("Anthropic request failed: %v", err)
		return
	}
	fmt.Printf("Anthropic -> %s\n", resp.Content)
}

func manualRouting(clients map[string]*litellm.Client) {
	ctx := context.Background()
	fmt.Println("\n2) Manual routing (business rules)")

	model := "gpt-4o-mini"
	provider := selectProviderByRule(model)
	client := clients[provider]
	if client == nil {
		log.Printf("Provider not found: %s", provider)
		return
	}

	resp, err := client.Chat(ctx, &litellm.Request{
		Model:     model,
		Messages:  []litellm.Message{{Role: "user", Content: "Manual routing"}},
		MaxTokens: litellm.IntPtr(50),
	})
	if err != nil {
		log.Printf("Manual routing request failed: %v", err)
		return
	}
	fmt.Printf("%s -> %s\n", provider, resp.Content)
}

// selectProviderByRule selects a provider using explicit, auditable rules.
func selectProviderByRule(model string) string {
	switch model {
	case "gpt-4o-mini", "gpt-5", "gpt-5.1":
		return "openai"
	case "claude-sonnet-4-5-20250929":
		return "anthropic"
	default:
		return "openai"
	}
}
