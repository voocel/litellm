package providers

import (
	"bufio"
	"encoding/json"
	"strings"
	"testing"
)

func TestMiniMaxDefaultBaseURL(t *testing.T) {
	p := NewMiniMax(ProviderConfig{})

	if got := p.Config().BaseURL; got != "https://api.minimax.io/v1" {
		t.Fatalf("base URL = %q, want %q", got, "https://api.minimax.io/v1")
	}
}

func TestMiniMaxThinkingMapper(t *testing.T) {
	p := NewMiniMax(ProviderConfig{})
	maxTokens := 100
	body, err := p.buildRequestBody(&Request{
		Model:     "MiniMax-M3",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: &maxTokens,
		Thinking:  &ThinkingConfig{Type: "enabled"},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	thinking, ok := got["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking missing or wrong type: %+v", got)
	}
	if thinking["type"] != "adaptive" {
		t.Fatalf("thinking.type = %#v, want adaptive; body=%+v", thinking["type"], got)
	}
	if got["reasoning_split"] != true {
		t.Fatalf("reasoning_split = %#v, want true; body=%+v", got["reasoning_split"], got)
	}
	if got["max_completion_tokens"] != float64(100) {
		t.Fatalf("max_completion_tokens = %#v, want 100; body=%+v", got["max_completion_tokens"], got)
	}
	if _, ok := got["max_tokens"]; ok {
		t.Fatalf("max_tokens should be omitted; body=%+v", got)
	}
}

func TestMiniMaxDisabledThinkingMapper(t *testing.T) {
	p := NewMiniMax(ProviderConfig{})
	body, err := p.buildRequestBody(&Request{
		Model:    "MiniMax-M3",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Thinking: &ThinkingConfig{Type: "disabled"},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	thinking, ok := got["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking missing or wrong type: %+v", got)
	}
	if thinking["type"] != "disabled" {
		t.Fatalf("thinking.type = %#v, want disabled; body=%+v", thinking["type"], got)
	}
	if _, ok := got["reasoning_split"]; ok {
		t.Fatalf("reasoning_split should be omitted when thinking is disabled: %+v", got)
	}
}

func TestMiniMaxExtractsReasoningDetails(t *testing.T) {
	p := NewMiniMax(ProviderConfig{})
	resp, err := p.convertResponse(&compatResponse{
		Model: "MiniMax-M3",
		Choices: []compatChoice{{
			Message: json.RawMessage(`{
				"role": "assistant",
				"content": "answer",
				"reasoning_details": [
					{"type": "reasoning.text", "text": "step 1"},
					{"type": "reasoning.text", "text": "step 2"}
				]
			}`),
			FinishReason: "stop",
		}},
	}, &Request{
		Model:    "MiniMax-M3",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Thinking: &ThinkingConfig{Type: "enabled"},
	})
	if err != nil {
		t.Fatalf("convertResponse failed: %v", err)
	}
	if resp.ReasoningContent != "step 1\n\nstep 2" {
		t.Fatalf("reasoning = %q, want concatenated reasoning details", resp.ReasoningContent)
	}
	if len(resp.ReasoningDetails) != 2 {
		t.Fatalf("reasoning_details len = %d, want 2", len(resp.ReasoningDetails))
	}
	if resp.ReasoningDetails[0]["text"] != "step 1" {
		t.Fatalf("reasoning_details[0].text = %#v", resp.ReasoningDetails[0]["text"])
	}
}

func TestMiniMaxPreservesReasoningDetailsInHistory(t *testing.T) {
	p := NewMiniMax(ProviderConfig{})
	body, err := p.buildRequestBody(&Request{
		Model: "MiniMax-M3",
		Messages: []Message{{
			Role:    "assistant",
			Content: "answer",
			ReasoningDetails: []map[string]any{
				{"type": "reasoning.text", "text": "step 1"},
			},
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildRequestBody failed: %v", err)
	}

	var got struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(got.Messages))
	}
	details, ok := got.Messages[0]["reasoning_details"].([]any)
	if !ok || len(details) != 1 {
		t.Fatalf("reasoning_details missing or wrong type: %+v", got.Messages[0])
	}
}

func TestMiniMaxStreamReasoningDetailsAreCumulative(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"model":"MiniMax-M3","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"step 1"}]}}]}`,
		`data: {"model":"MiniMax-M3","choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"step 1\nstep 2"}]}}]}`,
		`data: [DONE]`,
	}, "\n")

	p := NewMiniMax(ProviderConfig{})
	r := &compatStreamReader{
		scanner:          bufio.NewScanner(strings.NewReader(sse)),
		compat:           &p.compat,
		includeReasoning: true,
		model:            "MiniMax-M3",
	}

	var chunks []string
	for {
		chunk, err := r.Next()
		if err != nil {
			t.Fatalf("Next failed: %v", err)
		}
		if chunk.Done {
			break
		}
		if chunk.ReasoningContent != "" {
			chunks = append(chunks, chunk.ReasoningContent)
		}
	}

	if len(chunks) != 2 {
		t.Fatalf("reasoning chunk count = %d, want 2: %+v", len(chunks), chunks)
	}
	if chunks[0] != "step 1" || chunks[1] != "\nstep 2" {
		t.Fatalf("reasoning chunks = %+v, want cumulative suffixes", chunks)
	}
}
