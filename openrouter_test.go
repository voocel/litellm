package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestOpenRouter_BasicChat tests basic chat completion with OpenRouter
func TestOpenRouter_BasicChat(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}

	client := New(WithOpenRouter(apiKey))

	response, err := client.Complete(context.Background(), &Request{
		Model: "anthropic/claude-3.7-sonnet",
		Messages: []Message{
			{Role: "user", Content: "Hello! Please introduce yourself briefly."},
		},
		MaxTokens:   IntPtr(100),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	if response.Usage.TotalTokens == 0 {
		t.Error("Expected non-zero token usage")
	}

	t.Logf("Response: %s", response.Content)
	t.Logf("Usage: %+v", response.Usage)
}

// TestOpenRouter_ReasoningModel tests reasoning capabilities with OpenRouter
func TestOpenRouter_ReasoningModel(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}

	client := New(WithOpenRouter(apiKey))

	response, err := client.Complete(context.Background(), &Request{
		Model: "anthropic/claude-3.7-sonnet",
		Messages: []Message{
			{Role: "user", Content: "who are you?"},
		},
		MaxTokens:        IntPtr(2048),
		Temperature:      Float64Ptr(0.1),
		ReasoningEffort:  "medium",
		ReasoningSummary: "Show your work",
		UseResponsesAPI:  true,
	})

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	// Check if reasoning was provided
	if response.Reasoning != nil {
		t.Logf("Reasoning provided: %s", response.Reasoning.Summary)
		if response.Reasoning.Content == "" && response.Reasoning.Summary == "" {
			t.Error("Expected reasoning content or summary")
		}
	} else {
		t.Log("No reasoning data returned (this may be expected for some models)")
	}

	t.Logf("Response: %s", response.Content)
	t.Logf("Usage: %+v", response.Usage)
}

// TestOpenRouter_Streaming tests streaming responses
func TestOpenRouter_Streaming(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}

	client := New(WithOpenRouter(apiKey))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "anthropic/claude-3.5-sonnet",
		Messages: []Message{
			{Role: "user", Content: "Count from 1 to 5, one number per line."},
		},
		MaxTokens:   IntPtr(100),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("Stream request failed: %v", err)
	}
	defer stream.Close()

	var content strings.Builder
	chunkCount := 0

	for {
		chunk, err := stream.Read()
		if err != nil {
			t.Fatalf("Stream read failed: %v", err)
		}

		if chunk.Done {
			break
		}

		if chunk.Type == ChunkTypeContent && chunk.Content != "" {
			content.WriteString(chunk.Content)
			chunkCount++
		}

		// Prevent infinite loop
		if chunkCount > 100 {
			t.Error("Too many chunks received, possible infinite loop")
			break
		}
	}

	if chunkCount == 0 {
		t.Error("Expected to receive content chunks")
	}

	if content.Len() == 0 {
		t.Error("Expected non-empty streamed content")
	}

	t.Logf("Received %d chunks", chunkCount)
	t.Logf("Streamed content: %s", content.String())
}

// TestOpenRouter_ModelValidation tests that OpenRouter accepts various model names
func TestOpenRouter_ModelValidation(t *testing.T) {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENROUTER_API_KEY not set")
	}

	client := New(WithOpenRouter(apiKey))

	// Test with a model that might not be in the predefined list
	response, err := client.Complete(context.Background(), &Request{
		Model: "openai/gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "Say 'Hello OpenRouter!'"},
		},
		MaxTokens:   IntPtr(50),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("Request with gpt-4o-mini failed: %v", err)
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	if !strings.Contains(strings.ToLower(response.Content), "hello") {
		t.Errorf("Expected response to contain 'hello', got: %s", response.Content)
	}

	t.Logf("Response: %s", response.Content)
}

// TestOpenRouter_ProviderRegistration tests that OpenRouter provider is properly registered
func TestOpenRouter_ProviderRegistration(t *testing.T) {
	// Test that the provider factory is registered
	factory, exists := ProviderRegistry["openrouter"]
	if !exists {
		t.Error("OpenRouter provider not registered in ProviderRegistry")
	}

	// Test that we can create a provider instance
	config := ProviderConfig{
		APIKey:  "test-key",
		BaseURL: "https://openrouter.ai/api/v1",
	}

	provider := factory(config)
	if provider == nil {
		t.Error("Failed to create OpenRouter provider instance")
	}

	// Test that it implements the Provider interface (provider is already Provider type)
	// This is guaranteed by the factory function return type

	// Test validation (should pass with non-empty key)
	err := provider.Validate()
	if err != nil {
		t.Errorf("Expected validation to pass with non-empty key, got: %v", err)
	}

	// Test validation with empty key (should fail)
	emptyConfig := ProviderConfig{APIKey: ""}
	emptyProvider := factory(emptyConfig)
	err = emptyProvider.Validate()
	if err == nil {
		t.Error("Expected validation to fail with empty API key")
	}
}
