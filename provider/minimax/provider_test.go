package minimax

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/provider/compat"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestThinkingAndMaxCompletionTokens(t *testing.T) {
	maxTokens := 128
	body := captureBody(t, &litellm.Request{
		Model:     "MiniMax-M3",
		Messages:  []litellm.Message{litellm.UserText("hi")},
		MaxTokens: &maxTokens,
		Thinking:  &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if body["max_completion_tokens"] != float64(128) {
		t.Fatalf("max_completion_tokens = %#v", body["max_completion_tokens"])
	}
	if _, ok := body["max_tokens"]; ok {
		t.Fatalf("max_tokens should be omitted: %#v", body)
	}
	if thinking := body["thinking"].(map[string]any); thinking["type"] != "adaptive" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if body["reasoning_split"] != true {
		t.Fatalf("reasoning_split = %#v", body["reasoning_split"])
	}
	testgolden.AssertJSON(t, "../../testdata/compat/minimax_request.golden.json", body)
}

func TestDisabledThinking(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "MiniMax-M3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	if thinking := body["thinking"].(map[string]any); thinking["type"] != "disabled" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if _, ok := body["reasoning_split"]; ok {
		t.Fatalf("reasoning_split should be omitted: %#v", body)
	}
}

func captureBody(t *testing.T, req *litellm.Request) map[string]any {
	t.Helper()
	var body map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://minimax.test",
		HTTPClient: roundTripFunc(func(httpReq *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(httpReq.Body).Decode(&body); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"ok"}}]}`)), Header: make(http.Header)}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	return body
}
