package litellm

import (
	"context"
	"testing"
)

// Basic
func TestBasicUsage(t *testing.T) {
	client := New(WithOpenAI("sk-xxx", "xxx"))

	response, err := client.Complete(context.Background(), &Request{
		Model: "gpt-4.1",
		Messages: []Message{
			{Role: "user", Content: "who are you"},
		},
	})

	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	if response.Usage.TotalTokens == 0 {
		t.Error("Expected non-zero token usage")
	}

	t.Logf("Response: %s", response.Content)
	t.Logf("Tokens: %d", response.Usage.TotalTokens)
}

// Quick
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

// Multiple Provider
func TestMultipleProviders(t *testing.T) {
	client := New(
		WithOpenAI("test-key"),
		WithAnthropic("test-key"),
		WithGemini("test-key"),
	)

	providers := client.Providers()
	expected := []string{"openai", "anthropic", "gemini"}

	if len(providers) != len(expected) {
		t.Errorf("Expected %d providers, got %d", len(expected), len(providers))
	}

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

	t.Logf("Providers: %v", providers)
}

// reasoning model
func TestReasoningModel(t *testing.T) {
	client := New(WithOpenAI("sk-xxx", "https://one.wisehood.ai"))

	response, err := client.Complete(context.Background(), &Request{
		Model: "o1-mini",
		Messages: []Message{
			{Role: "user", Content: "2+2=？"},
		},
		ReasoningEffort:  "low",
		ReasoningSummary: "concise",
	})

	if err != nil {
		t.Logf("Reasoning model test failed (expected): %v", err)
		return
	}

	if response.Content == "" {
		t.Error("Expected non-empty response content")
	}

	t.Logf("Reasoning response: %s", response.Content)
	if response.Reasoning != nil {
		t.Logf("Reasoning summary: %s", response.Reasoning.Summary)
	}
}

// Function calling
func TestFunctionCalling(t *testing.T) {
	client := New(WithOpenAI("sk-xxx", "xxx"))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "get_weather",
				Description: "Get weather information",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type":        "string",
							"description": "city",
						},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	response, err := client.Complete(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "What's the weather like in New York？"},
		},
		Tools:      tools,
		ToolChoice: "auto",
	})

	if err != nil {
		t.Fatalf("Function calling failed: %v", err)
	}

	t.Logf("Response: %s", response.Content)
	if len(response.ToolCalls) > 0 {
		t.Logf("Tool called: %s", response.ToolCalls[0].Function.Name)
	}
}

// Streaming
func TestStreaming(t *testing.T) {
	client := New(WithOpenAI("sk-u5NLmQaEM0FDTOJg174aCb653c5d474c95D8Af5d77732645", "https://one.wisehood.ai"))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "tell me a joke"},
		},
	})

	if err != nil {
		t.Fatalf("Streaming failed: %v", err)
	}
	defer stream.Close()

	var content string
	chunkCount := 0

	for {
		chunk, err := stream.Read()
		if err != nil {
			t.Fatalf("Stream read failed: %v", err)
		}

		if chunk.Done {
			break
		}

		if chunk.Type == ChunkTypeContent {
			content += chunk.Content
			chunkCount++
		}
	}

	if content == "" {
		t.Error("Expected non-empty streamed content")
	}

	if chunkCount == 0 {
		t.Error("Expected at least one content chunk")
	}

	t.Logf("Streamed content: %s", content)
	t.Logf("Chunks received: %d", chunkCount)
}
