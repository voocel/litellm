package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestAnthropic_BasicChat tests basic chat completion
func TestAnthropic_BasicChat(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	client := New(WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "claude-4-sonnet",
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

// TestAnthropic_SystemMessage tests system message handling
func TestAnthropic_SystemMessage(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	client := New(WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "claude-4-sonnet",
		Messages: []Message{
			{Role: "system", Content: "You are a helpful math tutor. Always show your work step by step."},
			{Role: "user", Content: "What is 12 + 8?"},
		},
		MaxTokens:   IntPtr(150),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("System message request failed: %v", err)
	}

	t.Logf("✅ System Message")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)
	t.Logf("   Tokens: %d", response.Usage.TotalTokens)
}

// TestAnthropic_Streaming tests streaming responses with thinking
func TestAnthropic_Streaming(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	client := New(WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "claude-4-sonnet",
		Messages: []Message{
			{Role: "user", Content: "Solve this step by step: A train travels 120 km in 2 hours. What's its average speed? Show your thinking process."},
		},
		MaxTokens:   IntPtr(300),
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

// TestAnthropic_FunctionCalling tests function calling
func TestAnthropic_FunctionCalling(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	client := New(WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")))

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
		Model: "claude-4-sonnet",
		Messages: []Message{
			{Role: "user", Content: "What's the weather like in Paris? Use celsius."},
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

// TestAnthropic_MultiTurnConversation tests multi-turn conversation with tool calls
func TestAnthropic_MultiTurnConversation(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	client := New(WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")))

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

	// First turn: Ask for calculation
	response1, err := client.Complete(context.Background(), &Request{
		Model: "claude-4-sonnet",
		Messages: []Message{
			{Role: "user", Content: "Calculate 15 * 8 for me."},
		},
		Tools:       tools,
		ToolChoice:  "auto",
		MaxTokens:   IntPtr(150),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("First turn failed: %v", err)
	}

	t.Logf("✅ Multi-turn Conversation")
	t.Logf("   Turn 1 - Content: %s", response1.Content)

	if len(response1.ToolCalls) > 0 {
		t.Logf("   Turn 1 - Tool Call: %s(%s)",
			response1.ToolCalls[0].Function.Name,
			response1.ToolCalls[0].Function.Arguments)

		// Simulate tool execution result
		messages := []Message{
			{Role: "user", Content: "Calculate 15 * 8 for me."},
			{Role: "assistant", Content: response1.Content, ToolCalls: response1.ToolCalls},
			{Role: "tool", Content: "120", ToolCallID: response1.ToolCalls[0].ID},
		}

		// Second turn: Continue conversation with tool result
		response2, err := client.Complete(context.Background(), &Request{
			Model:       "claude-4-sonnet",
			Messages:    messages,
			MaxTokens:   IntPtr(100),
			Temperature: Float64Ptr(0.3),
		})

		if err != nil {
			t.Fatalf("Second turn failed: %v", err)
		}

		t.Logf("   Turn 2 - Content: %s", response2.Content)
	}
}
