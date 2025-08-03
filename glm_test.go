package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestGLM_BasicChat tests basic GLM conversation
func TestGLM_BasicChat(t *testing.T) {
	if os.Getenv("GLM_API_KEY") == "" {
		t.Skip("GLM_API_KEY not set")
	}

	client := New(WithGLM(os.Getenv("GLM_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "glm-4.5",
		Messages: []Message{
			{Role: "user", Content: "who are you?"},
		},
		MaxTokens:   IntPtr(500),
		Temperature: Float64Ptr(0.7),
		Extra: map[string]interface{}{
			"thinking": map[string]string{
				"type": "disabled",
			},
		},
	})

	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	t.Logf("✅ Basic Chat")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Provider: %s", response.Provider)
	t.Logf("   Content: %s", response.Content)
	if response.Reasoning != nil {
		t.Logf("   Reasoning: %s", response.Reasoning.Content)
	} else {
		t.Logf("   Reasoning: <none>")
	}
	t.Logf("   Tokens: %d (prompt: %d, completion: %d)",
		response.Usage.TotalTokens, response.Usage.PromptTokens, response.Usage.CompletionTokens)
}

// TestGLM_Streaming tests streaming responses
func TestGLM_Streaming(t *testing.T) {
	if os.Getenv("GLM_API_KEY") == "" {
		t.Skip("GLM_API_KEY not set")
	}

	client := New(WithGLM(os.Getenv("GLM_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "glm-4.5",
		Messages: []Message{
			{Role: "user", Content: "who are you。"},
		},
		MaxTokens:   IntPtr(500),
		Temperature: Float64Ptr(0.7),
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

// TestGLM_FunctionCalling tests function calling
func TestGLM_FunctionCalling(t *testing.T) {
	if os.Getenv("GLM_API_KEY") == "" {
		t.Skip("GLM_API_KEY not set")
	}

	client := New(WithGLM(os.Getenv("GLM_API_KEY")))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "get_weather",
				Description: "Get weather information for a specified city",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{
							"type":        "string",
							"description": "City Name",
						},
						"unit": map[string]interface{}{
							"type": "string",
							"enum": []string{"celsius", "fahrenheit"},
						},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	response, err := client.Complete(context.Background(), &Request{
		Model: "glm-4.5",
		Messages: []Message{
			{Role: "user", Content: "What's the weather like in Beijing today?"},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   IntPtr(500),
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

// TestGLM_ThinkingMode tests GLM-4.5 thinking mode
func TestGLM_ThinkingMode(t *testing.T) {
	if os.Getenv("GLM_API_KEY") == "" {
		t.Skip("GLM_API_KEY not set")
	}

	client := New(WithGLM(os.Getenv("GLM_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "glm-4.5",
		Messages: []Message{
			{Role: "user", Content: "who are you?"},
		},
		MaxTokens:   IntPtr(500),
		Temperature: Float64Ptr(0.3),
		Extra: map[string]interface{}{
			"thinking": map[string]string{
				"type": "enabled",
			},
		},
	})

	if err != nil {
		t.Fatalf("Thinking mode request failed: %v", err)
	}

	t.Logf("✅ Thinking Mode")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)
	if response.Reasoning != nil {
		t.Logf("   Reasoning: %s", response.Reasoning.Content)
		t.Logf("   Reasoning Tokens: %d", response.Reasoning.TokensUsed)
	}
}
