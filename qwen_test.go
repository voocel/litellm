package litellm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// TestQwen_BasicChat tests basic chat completion with Qwen3-Coder-Plus
func TestQwen_BasicChat(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "qwen3-coder-plus",
		Messages: []Message{
			{Role: "system", Content: "You are a helpful coding assistant."},
			{Role: "user", Content: "Write a simple Python function to calculate factorial. Keep it brief."},
		},
		MaxTokens:   IntPtr(200),
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

	// Verify response contains code-related content
	if !strings.Contains(strings.ToLower(response.Content), "def") &&
		!strings.Contains(strings.ToLower(response.Content), "factorial") {
		t.Logf("   Warning: Response may not contain expected code content")
	}
}

// TestQwen_CodeGeneration tests code generation with different models
func TestQwen_CodeGeneration(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	// Test with Qwen3-Coder-Flash for faster response
	response, err := client.Complete(context.Background(), &Request{
		Model: "qwen3-coder-flash",
		Messages: []Message{
			{Role: "user", Content: "Create a Go function that implements binary search. Include comments."},
		},
		MaxTokens:   IntPtr(300),
		Temperature: Float64Ptr(0.3),
	})

	if err != nil {
		t.Fatalf("Code generation request failed: %v", err)
	}

	t.Logf("✅ Code Generation")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)
	t.Logf("   Tokens: %d", response.Usage.TotalTokens)

	// Verify response contains Go code
	if !strings.Contains(response.Content, "func") ||
		!strings.Contains(strings.ToLower(response.Content), "binary") {
		t.Logf("   Warning: Response may not contain expected Go code")
	}
}

// TestQwen_Streaming tests streaming responses
func TestQwen_Streaming(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "qwen3-coder-plus",
		Messages: []Message{
			{Role: "user", Content: "Explain the differences between synchronous and asynchronous programming. Keep it concise."},
		},
		MaxTokens:   IntPtr(250),
		Temperature: Float64Ptr(0.8),
	})

	if err != nil {
		t.Fatalf("Streaming request failed: %v", err)
	}
	defer stream.Close()

	t.Logf("✅ Streaming Response")

	var content strings.Builder
	contentChunks := 0

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
		default:
			if chunk.Content != "" {
				t.Logf("Other (%s): %s", chunk.Type, chunk.Content)
			}
		}
	}

	t.Logf("")
	t.Logf("Content chunks: %d", contentChunks)
	t.Logf("Final content: %s", content.String())

	if contentChunks == 0 {
		t.Errorf("No content chunks received")
	}
}

// TestQwen_FunctionCalling tests function calling capabilities
func TestQwen_FunctionCalling(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	tools := []Tool{
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "write_file",
				Description: "Write content to a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "The file path to write to",
						},
						"content": map[string]interface{}{
							"type":        "string",
							"description": "The content to write to the file",
						},
					},
					"required": []string{"path", "content"},
				},
			},
		},
		{
			Type: "function",
			Function: FunctionSchema{
				Name:        "read_file",
				Description: "Read content from a file",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path": map[string]interface{}{
							"type":        "string",
							"description": "The file path to read from",
						},
					},
					"required": []string{"path"},
				},
			},
		},
	}

	response, err := client.Complete(context.Background(), &Request{
		Model: "qwen3-coder-plus",
		Messages: []Message{
			{Role: "user", Content: "Create a simple Python hello world program and save it to hello.py"},
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

// TestQwen_MultiTurnConversation tests multi-turn conversation
func TestQwen_MultiTurnConversation(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	// First turn
	messages := []Message{
		{Role: "user", Content: "I need help with a Python web scraping project"},
	}

	response1, err := client.Complete(context.Background(), &Request{
		Model:       "qwen3-coder-plus",
		Messages:    messages,
		MaxTokens:   IntPtr(150),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("First turn failed: %v", err)
	}

	t.Logf("✅ Multi-turn Conversation")
	t.Logf("   Turn 1 - User: %s", messages[0].Content)
	t.Logf("   Turn 1 - Assistant: %s", response1.Content)

	// Add assistant response to conversation
	messages = append(messages, Message{Role: "assistant", Content: response1.Content})

	// Second turn
	messages = append(messages, Message{Role: "user", Content: "I want to scrape product information from an e-commerce site. What libraries should I use?"})

	response2, err := client.Complete(context.Background(), &Request{
		Model:       "qwen3-coder-plus",
		Messages:    messages,
		MaxTokens:   IntPtr(200),
		Temperature: Float64Ptr(0.7),
	})

	if err != nil {
		t.Fatalf("Second turn failed: %v", err)
	}

	t.Logf("   Turn 2 - User: %s", messages[2].Content)
	t.Logf("   Turn 2 - Assistant: %s", response2.Content)
	t.Logf("   Total tokens used: %d", response1.Usage.TotalTokens+response2.Usage.TotalTokens)
}

// TestQwen_ReasoningMode tests Qwen3's thinking capabilities
func TestQwen_ReasoningMode(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	response, err := client.Complete(context.Background(), &Request{
		Model: "qwen3-coder-plus",
		Messages: []Message{
			{Role: "user", Content: "请写一个Python函数来实现快速排序算法，并详细解释你的思路和每一步的原理。"},
		},
		MaxTokens:   IntPtr(1000),
		Temperature: Float64Ptr(0.1),
		Extra: map[string]interface{}{
			"enable_thinking": true,
		},
	})

	if err != nil {
		t.Fatalf("Reasoning request failed: %v", err)
	}

	t.Logf("✅ Reasoning Mode")
	t.Logf("   Model: %s", response.Model)
	t.Logf("   Content: %s", response.Content)
	t.Logf("   Tokens: %d", response.Usage.TotalTokens)

	if response.Reasoning != nil {
		t.Logf("   ✅ Reasoning Content Found: %s", response.Reasoning.Content)
		t.Logf("   Reasoning Summary: %s", response.Reasoning.Summary)
		t.Logf("   Reasoning Tokens: %d", response.Reasoning.TokensUsed)
	} else {
		t.Logf("   ❌ No reasoning data available")
		t.Logf("   This could mean:")
		t.Logf("   - The model doesn't support thinking mode")
		t.Logf("   - The question was too simple to require thinking")
		t.Logf("   - The API response format is different than expected")
	}

	// Verify response contains code-related content
	if !strings.Contains(strings.ToLower(response.Content), "def") &&
		!strings.Contains(strings.ToLower(response.Content), "function") &&
		!strings.Contains(strings.ToLower(response.Content), "排序") {
		t.Logf("   Warning: Response may not contain expected code content")
	}
}

// TestQwen_StreamingReasoning tests streaming responses with thinking
func TestQwen_StreamingReasoning(t *testing.T) {
	if os.Getenv("QWEN_API_KEY") == "" {
		t.Skip("QWEN_API_KEY not set")
	}
	client := New(WithQwen(os.Getenv("QWEN_API_KEY")))

	stream, err := client.Stream(context.Background(), &Request{
		Model: "qwen3-coder-plus",
		Messages: []Message{
			{Role: "user", Content: "Think through this problem: How would you implement a binary search tree in Go? Show your reasoning process."},
		},
		MaxTokens:   IntPtr(400),
		Temperature: Float64Ptr(0.3),
		Extra: map[string]interface{}{
			"enable_thinking": true,
		},
	})

	if err != nil {
		t.Fatalf("Streaming reasoning request failed: %v", err)
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
			if chunk.Reasoning != nil && chunk.Reasoning.Content != "" {
				reasoning.WriteString(chunk.Reasoning.Content)
				t.Logf("Thinking: %s", chunk.Reasoning.Content)
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
		t.Logf("Final reasoning: %s", reasoning.String())
	}
}

// TestQwen_Validate tests provider validation
func TestQwen_Validate(t *testing.T) {
	// Test with valid API key
	provider := NewQwenProvider(ProviderConfig{APIKey: "sk-test-key"})
	if err := provider.Validate(); err != nil {
		t.Errorf("Validation failed with valid API key: %v", err)
	}

	// Test with empty API key
	provider = NewQwenProvider(ProviderConfig{APIKey: ""})
	if err := provider.Validate(); err == nil {
		t.Error("Validation should fail with empty API key")
	}

	t.Logf("✅ Provider Validation")
}
