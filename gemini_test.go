package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestGemini_BasicChat tests basic Gemini conversation
func TestGemini_BasicChat(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	client := New(WithGemini(os.Getenv("GEMINI_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "gemini-2.5-flash",
		Messages: []Message{
			{Role: "user", Content: "Hello! Please introduce yourself briefly."},
		},
		MaxTokens:   IntPtr(100),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	t.Logf("✅ Basic Chat")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Provider: %s", response.Provider)
	t.Logf("   Content: %s", response.Content)
	t.Logf("   Tokens: %d (prompt: %d, completion: %d)",
		response.Usage.TotalTokens, response.Usage.PromptTokens, response.Usage.CompletionTokens)
}

// TestGemini_Streaming tests streaming responses
func TestGemini_Streaming(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	client := New(WithGemini(os.Getenv("GEMINI_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "gemini-2.5-flash",
		Messages: []Message{
			{Role: "user", Content: "Explain how neural networks work in simple terms."},
		},
		MaxTokens:   IntPtr(300),
		Temperature: Float64Ptr(0.6),
	})

	if err != nil {
		t.Fatalf("Streaming request failed: %v", err)
	}
	defer stream.Close()

	t.Logf("✅ Streaming Response")

	var content strings.Builder
	var reasoning strings.Builder
	contentChunks := 0
	reasoningChunks := 0

	for {
		chunk, err := stream.Read()
		if err != nil || chunk.Done {
			break
		}

		switch chunk.Type {
		case ChunkTypeContent:
			if chunk.Content != "" {
				content.WriteString(chunk.Content)
				t.Logf("Content: %s", chunk.Content)
				contentChunks++
			}
		case ChunkTypeReasoning:
			if chunk.Content != "" {
				reasoning.WriteString(chunk.Content)
				t.Logf("Thinking: %s", chunk.Content)
				reasoningChunks++
			}
		default:
			if chunk.Content != "" {
				t.Logf("Other (%s): %s", chunk.Type, chunk.Content)
			}
		}
	}

	t.Logf("")
	t.Logf("Content chunks: %d", contentChunks)
	t.Logf("Reasoning chunks: %d", reasoningChunks)
	t.Logf("Final content: %s", content.String())
	if reasoning.Len() > 0 {
		t.Logf("   Final reasoning: %s", reasoning.String())
	}
}

// TestGemini_FunctionCalling tests function calling
func TestGemini_FunctionCalling(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	client := New(WithGemini(os.Getenv("GEMINI_API_KEY")))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "search_knowledge",
				Description: "Search for information in a knowledge base",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "Search query",
						},
						"category": map[string]interface{}{
							"type": "string",
							"enum": []string{"science", "technology", "history", "general"},
						},
					},
					"required": []string{"query"},
				},
			},
		},
	}

	response, err := client.Complete(context.Background(), &Request{
		Model: "gemini-2.5-pro",
		Messages: []Message{
			{Role: "user", Content: "Search for information about quantum computing"},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   IntPtr(200),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("Function calling request failed: %v", err)
	}

	t.Logf("✅ Function Calling")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)

	if len(response.ToolCalls) > 0 {
		for i, toolCall := range response.ToolCalls {
			t.Logf("   Tool Call %d:", i+1)
			t.Logf("     ID: %s", toolCall.ID)
			t.Logf("     Type: %s", toolCall.Type)
			t.Logf("     Function: %s", toolCall.Function.Name)
			t.Logf("     Arguments: %s", toolCall.Function.Arguments)
		}
	} else {
		t.Logf("   No tool calls made - model responded directly")
	}
}

// TestGemini_MultiTurnConversation tests multi-turn conversations
func TestGemini_MultiTurnConversation(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	client := New(WithGemini(os.Getenv("GEMINI_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "gemini-2.5-flash",
		Messages: []Message{
			{Role: "user", Content: "What is machine learning?"},
			{Role: "assistant", Content: "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
			{Role: "user", Content: "Can you give me a simple example?"},
		},
		MaxTokens:   IntPtr(150),
		Temperature: Float64Ptr(0.5),
	})

	if err != nil {
		t.Fatalf("Multi-turn conversation failed: %v", err)
	}

	t.Logf("✅ Multi-turn Conversation")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)
	t.Logf("   Tokens: %d", response.Usage.TotalTokens)
}
