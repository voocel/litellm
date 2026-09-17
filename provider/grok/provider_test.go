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
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "high"},
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

func TestGrok45Reasoning(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.5-latest",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "medium"},
	})
	if body["reasoning_effort"] != "medium" {
		t.Fatalf("body = %#v", body)
	}

	disabled := captureBody(t, &litellm.Request{
		Model:    "grok-4.5",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	if disabled["reasoning_effort"] != "none" {
		t.Fatalf("reasoning_effort = %#v, want none", disabled["reasoning_effort"])
	}
}

func TestToolsAreAlwaysStrict(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.5",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Tools:    []litellm.Tool{mustTool(t, "lookup", litellm.StrictEnabled)},
	})
	fn := body["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if _, ok := fn["strict"]; ok {
		t.Fatalf("xAI enforces strict tool schemas without a strict field: %#v", fn)
	}
}

func TestReasoningEffortPassesThroughForNewModels(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-next",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "xhigh"},
	})
	if body["reasoning_effort"] != "xhigh" {
		t.Fatalf("reasoning_effort = %#v, want xhigh", body["reasoning_effort"])
	}
}

func TestThinkingUsesDefaultEffort(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v, want high", body["reasoning_effort"])
	}
}

func TestGrok46SupportsXHigh(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "grok-4.6",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "xhigh"},
	})
	if body["reasoning_effort"] != "xhigh" {
		t.Fatalf("reasoning_effort = %#v, want xhigh", body["reasoning_effort"])
	}
}

func TestRejectsUnsupportedReasoningEffort(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://grok.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "max"},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported reasoning_effort") {
		t.Fatalf("expected effort error, got %v", err)
	}
}

func TestRejectsStopForReasoningModel(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://grok.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "grok-4.3",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Stop:     []string{"END"},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err == nil || !strings.Contains(err.Error(), "stop is not supported") {
		t.Fatalf("expected stop error, got %v", err)
	}
}

func TestRejectsUnsupportedReasoningProviderOptions(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://grok.test", HTTPClient: roundTripFunc(nil), AllowUnknownProviderOptions: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:           "grok-4.3",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		Thinking:        &litellm.Thinking{Mode: litellm.ThinkingEnabled},
		ProviderOptions: mustProviderOptions(t, map[string]any{"presence_penalty": 0.2}),
	})
	if err == nil || !strings.Contains(err.Error(), "presence_penalty") {
		t.Fatalf("expected provider option error, got %v", err)
	}
}

func TestCapabilities(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://grok.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	caps := p.Capabilities("grok-4.3")
	if caps.Thinking.Supported != litellm.SupportPartial || caps.Thinking.Disable != litellm.SupportPartial || !caps.Thinking.SupportsEffort("high") || caps.Thinking.SupportsEffort("xhigh") {
		t.Fatalf("thinking caps = %+v", caps.Thinking)
	}
	if alias := p.Capabilities("grok-latest"); alias.Thinking.Supported != litellm.SupportPartial {
		t.Fatalf("alias thinking caps = %+v", alias.Thinking)
	}
	if latest := p.Capabilities("grok-4.5-latest"); latest.Thinking.Supported != litellm.SupportPartial || latest.Tools.StrictSchema != litellm.SupportYes {
		t.Fatalf("grok-4.5 caps = %+v", latest)
	}
	if future := p.Capabilities("grok-next"); future.Thinking.Supported != litellm.SupportPartial || !future.Thinking.SupportsEffort("high") {
		t.Fatalf("future model caps = %+v", future.Thinking)
	}
}

func mustTool(t *testing.T, name string, strict litellm.StrictMode) litellm.Tool {
	t.Helper()
	tool, err := litellm.NewTool(name, "Lookup.", map[string]any{"type": "object"})
	if err != nil {
		t.Fatalf("NewTool: %v", err)
	}
	tool.Strict = strict
	return tool
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

func mustProviderOptions(t *testing.T, values map[string]any) litellm.ProviderOptions {
	t.Helper()
	o, err := litellm.NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}
