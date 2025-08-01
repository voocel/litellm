package litellm

import (
	"testing"
)

// TestQuickMethod tests the Quick convenience method
func TestQuickMethod(t *testing.T) {
	// Note: This test requires a valid API key to run
	t.Skip("Skipping Quick method test - requires valid API key")

	response, err := Quick("gpt-4o-mini", "Hello, LiteLLM!")
	if err != nil {
		t.Fatalf("Quick method failed: %v", err)
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	t.Logf("Quick Response: %s", response.Content)
}

// TestMultipleProviders tests multiple provider configuration
func TestMultipleProviders(t *testing.T) {
	client := New(
		WithOpenAI("test-key"),
		WithAnthropic("test-key"),
		WithGemini("test-key"),
	)

	providers := client.Providers()
	expected := []string{"openai", "anthropic", "gemini"}

	// Check that all expected providers are present
	for _, exp := range expected {
		found := false
		for _, prov := range providers {
			if prov == exp {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Provider %s not found", exp)
		}
	}

	// Should have at least the expected providers
	if len(providers) < len(expected) {
		t.Errorf("Expected at least %d providers, got %d", len(expected), len(providers))
	}

	t.Logf("Providers: %v", providers)
}

// TestSimplifiedModelRouting tests the simplified model routing logic
func TestSimplifiedModelRouting(t *testing.T) {
	// Test single provider scenario - should work with any model
	t.Run("SingleProvider_AnyModel", func(t *testing.T) {
		// Create client without auto-discovery to ensure only one provider
		client := &Client{
			providers: make(map[string]Provider),
			defaults: DefaultConfig{
				MaxTokens:   4096,
				Temperature: 0.7,
			},
		}

		// Manually add only one provider
		WithOpenAI("test-key")(client)

		providers := client.Providers()
		if len(providers) != 1 {
			t.Fatalf("Expected exactly 1 provider, got %d", len(providers))
		}

		// Should work with any model name when only one provider is configured
		testModels := []string{
			"some-unknown-model",
			"gpt-5-ultra",             // Future model
			"custom-fine-tuned-model", // Custom model
			"o3-mini",                 // Known model
		}

		for _, model := range testModels {
			provider, err := client.resolveProvider(model)
			if err != nil {
				t.Errorf("Single provider should accept any model '%s', got error: %v", model, err)
				continue
			}
			if provider == nil {
				t.Errorf("Provider is nil for model '%s'", model)
				continue
			}
			if provider.Name() != "openai" {
				t.Errorf("Expected openai provider for model '%s', got %s", model, provider.Name())
			}
		}
	})

	// Test multiple providers - must match predefined models
	t.Run("MultipleProviders_PredefinedOnly", func(t *testing.T) {
		// Create client without auto-discovery
		client := &Client{
			providers: make(map[string]Provider),
			defaults: DefaultConfig{
				MaxTokens:   4096,
				Temperature: 0.7,
			},
		}

		// Manually add multiple providers
		WithOpenAI("test-key")(client)
		WithAnthropic("test-key")(client)

		// Test known models - should work
		knownModels := []struct {
			model    string
			expected string
		}{
			{"gpt-4o", "openai"},
			{"gpt-4o-mini", "openai"},
			{"o3-mini", "openai"},
		}

		for _, tc := range knownModels {
			provider, err := client.resolveProvider(tc.model)
			if err != nil {
				t.Errorf("Failed to resolve known model %s: %v", tc.model, err)
				continue
			}
			if provider.Name() != tc.expected {
				t.Errorf("Model %s: expected provider %s, got %s", tc.model, tc.expected, provider.Name())
			}
		}

		// Test unknown models - should fail
		unknownModels := []string{
			"some-unknown-model",
			"gpt-5-ultra",
			"custom-model",
		}

		for _, model := range unknownModels {
			_, err := client.resolveProvider(model)
			if err == nil {
				t.Errorf("Unknown model '%s' should fail with multiple providers, but succeeded", model)
			}
		}
	})
}
