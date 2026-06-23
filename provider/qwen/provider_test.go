package qwen

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

func TestThinkingBudget(t *testing.T) {
	budget := 4096
	body := captureBody(t, &litellm.Request{
		Model:    "qwen-plus",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, BudgetTokens: &budget},
	})
	if body["enable_thinking"] != true || body["thinking_budget"] != float64(4096) {
		t.Fatalf("body = %#v", body)
	}
	testgolden.AssertJSON(t, "../../testdata/compat/qwen_request.golden.json", body)
}

func TestThinkingDisabledErrors(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://qwen.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "qwen-plus",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	if err == nil || !strings.Contains(err.Error(), "disabling thinking is not supported") {
		t.Fatalf("expected disabled thinking error, got %v", err)
	}
}

func TestResponseReasoningContent(t *testing.T) {
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://qwen.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(strings.NewReader(`{
					"choices":[{"message":{"content":"ok","reasoning_content":"think"},"finish_reason":"stop"}]
				}`)),
				Header: make(http.Header),
			}, nil
		}),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := p.Chat(context.Background(), &litellm.Request{
		Model:    "qwen-plus",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Text() != "ok" || resp.Reasoning() != "think" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
}

func captureBody(t *testing.T, req *litellm.Request) map[string]any {
	t.Helper()
	var body map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://qwen.test",
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
