package providers

import "testing"

func TestPrepareOpenAIMessagesUsesPrepareMessages(t *testing.T) {
	messages := []Message{
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call.with.invalid/chars",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup_weather",
						Arguments: `{"city":"Paris"}`,
					},
				},
			},
		},
	}

	prepared, err := prepareOpenAIMessages(messages)
	if err != nil {
		t.Fatalf("prepareOpenAIMessages failed: %v", err)
	}
	if len(prepared) != 2 {
		t.Fatalf("prepared message count = %d, want 2", len(prepared))
	}
	if prepared[0].ToolCalls[0].ID != "call_with_invalid_chars" {
		t.Fatalf("tool call id = %q, want %q", prepared[0].ToolCalls[0].ID, "call_with_invalid_chars")
	}
	if prepared[1].Role != "tool" {
		t.Fatalf("synthetic message role = %q, want %q", prepared[1].Role, "tool")
	}
	if prepared[1].ToolCallID != "call_with_invalid_chars" {
		t.Fatalf("synthetic tool_call_id = %q, want %q", prepared[1].ToolCallID, "call_with_invalid_chars")
	}
}

func TestOpenAIResolveTokenParamsOnlySpecialCasesGPT5(t *testing.T) {
	p := NewOpenAI(ProviderConfig{APIKey: "test"})
	maxTokens := 123

	t.Run("gpt5 uses max_completion_tokens", func(t *testing.T) {
		max, maxCompletion := p.resolveTokenParams(&Request{
			Model:     "openai/gpt-5.5",
			MaxTokens: &maxTokens,
		})
		if max != nil {
			t.Fatalf("max_tokens = %v, want nil", *max)
		}
		if maxCompletion == nil || *maxCompletion != maxTokens {
			t.Fatalf("max_completion_tokens = %v, want %d", maxCompletion, maxTokens)
		}
	})

	t.Run("o series no longer uses max_completion_tokens", func(t *testing.T) {
		max, maxCompletion := p.resolveTokenParams(&Request{
			Model:     "o3-mini",
			MaxTokens: &maxTokens,
		})
		if max == nil || *max != maxTokens {
			t.Fatalf("max_tokens = %v, want %d", max, maxTokens)
		}
		if maxCompletion != nil {
			t.Fatalf("max_completion_tokens = %v, want nil", *maxCompletion)
		}
	})
}

func TestOpenAIChatExtraMapsOfficialFields(t *testing.T) {
	req := &openaiRequest{}
	err := applyOpenAIChatExtra(req, map[string]any{
		"frequency_penalty":      0.2,
		"presence_penalty":       0.3,
		"logit_bias":             map[string]any{"42": -10},
		"n":                      2,
		"logprobs":               true,
		"top_logprobs":           3,
		"store":                  true,
		"prompt_cache_key":       "chat-v1",
		"prompt_cache_retention": "24h",
		"prediction":             map[string]any{"type": "content", "content": "known output"},
		"metadata":               map[string]any{"tenant": "acme"},
		"modalities":             []any{"text"},
		"service_tier":           "flex",
		"user":                   "user-123",
		"seed":                   7,
	})
	if err != nil {
		t.Fatalf("applyOpenAIChatExtra: %v", err)
	}

	if req.FrequencyPenalty == nil || *req.FrequencyPenalty != 0.2 {
		t.Fatalf("frequency_penalty = %v", req.FrequencyPenalty)
	}
	if req.PresencePenalty == nil || *req.PresencePenalty != 0.3 {
		t.Fatalf("presence_penalty = %v", req.PresencePenalty)
	}
	if req.LogitBias["42"] != -10 {
		t.Fatalf("logit_bias = %#v", req.LogitBias)
	}
	if req.N == nil || *req.N != 2 || req.TopLogprobs == nil || *req.TopLogprobs != 3 {
		t.Fatalf("integer extras not mapped: %#v", req)
	}
	if req.Logprobs == nil || !*req.Logprobs || req.Store == nil || !*req.Store {
		t.Fatalf("bool extras not mapped: %#v", req)
	}
	if req.PromptCacheKey != "chat-v1" || req.PromptCacheRetention != "24h" {
		t.Fatalf("prompt cache fields not mapped: %#v", req)
	}
	if req.Prediction == nil || req.Prediction.Type != "content" || req.Prediction.Content != "known output" {
		t.Fatalf("prediction = %#v", req.Prediction)
	}
	if req.Metadata["tenant"] != "acme" || len(req.Modalities) != 1 || req.Modalities[0] != "text" {
		t.Fatalf("metadata/modalities not mapped: %#v", req)
	}
	if req.ServiceTier != "flex" || req.User != "user-123" || req.Seed == nil || *req.Seed != 7 {
		t.Fatalf("service/user/seed not mapped: %#v", req)
	}
}

func TestOpenAIChatExtraRejectsUnsupportedKeys(t *testing.T) {
	err := validateOpenAIChatExtra(map[string]any{"unknown": true})
	if err == nil {
		t.Fatal("expected unsupported extra error")
	}
}
