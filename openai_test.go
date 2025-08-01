package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestOpenAI_BasicChat tests basic chat completion
func TestOpenAI_BasicChat(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	client := New(WithOpenAI(os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_BASE_URL")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "Say hello and introduce yourself briefly."},
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

// TestOpenAI_ReasoningModel tests reasoning models (o-series)
func TestOpenAI_ReasoningModel(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	client := New(WithOpenAI(os.Getenv("OPENAI_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "o3-mini",
		Messages: []Message{
			{Role: "user", Content: "Solve this step by step: What is 15% of 240?"},
		},
		MaxTokens:        IntPtr(500),
		ReasoningEffort:  "medium",
		ReasoningSummary: "detailed",
	})

	if err != nil {
		t.Fatalf("Reasoning request failed: %v", err)
	}

	t.Logf("✅ Reasoning Model")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Answer: %s", response.Content)
	t.Logf("   Tokens: %d (reasoning: %d)", response.Usage.TotalTokens, response.Usage.ReasoningTokens)

	if response.Reasoning != nil {
		t.Logf("   Reasoning Content: %s", response.Reasoning.Content)
		t.Logf("   Reasoning Summary: %s", response.Reasoning.Summary)
		t.Logf("   Reasoning Tokens: %d", response.Reasoning.TokensUsed)
	} else {
		t.Logf("   No reasoning data available")
	}
}

// TestOpenAI_Streaming tests streaming responses with thinking
func TestOpenAI_Streaming(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	client := New(WithOpenAI(os.Getenv("OPENAI_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "Solve this math problem step by step: What is 25% of 80? Show your reasoning."},
		},
		MaxTokens:   IntPtr(200),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("Streaming request failed: %v", err)
	}
	defer stream.Close()

	t.Logf("✅ Streaming Response with Thinking")

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

// TestOpenAI_FunctionCalling tests function calling
func TestOpenAI_FunctionCalling(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	client := New(WithOpenAI(os.Getenv("OPENAI_API_KEY")))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "get_weather",
				Description: "Get current weather for a city",
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

	response, err := client.Complete(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "What's the weather in Tokyo? Use celsius."},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   IntPtr(150),
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

// TestOpenAI_StreamingToolCalls tests streaming function calls
func TestOpenAI_StreamingToolCalls(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	client := New(WithOpenAI(os.Getenv("OPENAI_API_KEY")))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "calculate",
				Description: "Perform mathematical calculations",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expression": map[string]interface{}{
							"type":        "string",
							"description": "Mathematical expression to calculate",
						},
					},
					"required": []string{"expression"},
				},
			},
		},
	}

	stream, err := client.Stream(context.Background(), &Request{
		Model: "gpt-4o-mini",
		Messages: []Message{
			{Role: "user", Content: "Calculate 25 * 4 using the calculate function."},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   IntPtr(150),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("Streaming tool calls request failed: %v", err)
	}
	defer stream.Close()

	t.Logf("✅ Streaming Tool Calls")

	var content strings.Builder
	toolCallDeltas := make(map[string]*strings.Builder)

	for {
		chunk, err := stream.Read()
		if err != nil || chunk.Done {
			break
		}

		switch chunk.Type {
		case ChunkTypeContent:
			if chunk.Content != "" {
				content.WriteString(chunk.Content)
				t.Logf("   Content: %s", chunk.Content)
			}
		case ChunkTypeToolCallDelta:
			if chunk.ToolCallDelta != nil && chunk.ToolCallDelta.ID != "" {
				if _, exists := toolCallDeltas[chunk.ToolCallDelta.ID]; !exists {
					toolCallDeltas[chunk.ToolCallDelta.ID] = &strings.Builder{}
					t.Logf("   Tool Call ID: %s", chunk.ToolCallDelta.ID)
				}
				if chunk.ToolCallDelta.ArgumentsDelta != "" {
					toolCallDeltas[chunk.ToolCallDelta.ID].WriteString(chunk.ToolCallDelta.ArgumentsDelta)
					t.Logf("   Args Delta: %s", chunk.ToolCallDelta.ArgumentsDelta)
				}
			}
		}
	}

	if len(toolCallDeltas) > 0 {
		for id, args := range toolCallDeltas {
			t.Logf("   Final Tool Call - ID: %s, Full Args: %s", id, args.String())
		}
	} else {
		t.Logf("   No tool calls - Direct response: %s", content.String())
	}
}
