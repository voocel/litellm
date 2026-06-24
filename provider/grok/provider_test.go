package grok

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

func TestReasoningEffort(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
	})
	if body["reasoning_effort"] != "high" {
		t.Fatalf("body = %#v", body)
	}
	testgolden.AssertJSON(t, "../../testdata/compat/grok_request.golden.json", body)
}

func TestThinkingDisabled(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	if body["reasoning_effort"] != "none" {
		t.Fatalf("body = %#v", body)
	}
}

func TestReasoningEffortDefersModelSupportToAPI(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
	})
	if body["reasoning_effort"] != "high" {
		t.Fatalf("body = %#v", body)
	}
}

func TestThinkingRequiresLevelOrEffort(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://grok.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err == nil || !strings.Contains(err.Error(), "level or effort is required") {
		t.Fatalf("expected thinking requirement error, got %v", err)
	}
}

func captureBody(t *testing.T, req *litellm.Request) map[string]any {
	t.Helper()
	var body map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://grok.test",
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
