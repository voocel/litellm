package glm

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

func TestThinkingVersionGates(t *testing.T) {
	body := captureBody(t, &litellm.Request{
		Model:    "glm-5.2",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "HIGH"},
	})
	if thinking := body["thinking"].(map[string]any); thinking["type"] != "enabled" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if body["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v", body["reasoning_effort"])
	}
	testgolden.AssertJSON(t, "../../testdata/compat/glm_request.golden.json", body)

	body = captureBody(t, &litellm.Request{
		Model:    "glm-5.1",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
	})
	if _, ok := body["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort should be omitted below 5.2: %#v", body)
	}
}

func TestThinkingUnsupportedModelErrors(t *testing.T) {
	p, err := New(compat.Config{APIKey: "key", BaseURL: "https://glm.test", HTTPClient: roundTripFunc(nil)})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:    "glm-4",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
	})
	if err == nil || !strings.Contains(err.Error(), "glm-4.5 or later") {
		t.Fatalf("expected unsupported thinking error, got %v", err)
	}
}

func TestJSONSchemaToPromptStillSendsJSONObjectFormat(t *testing.T) {
	format, err := litellm.NewResponseFormatJSONSchema("answer", "", map[string]any{
		"type":       "object",
		"properties": map[string]any{"ok": map[string]any{"type": "boolean"}},
	}, litellm.StrictDefault)
	if err != nil {
		t.Fatalf("NewResponseFormatJSONSchema: %v", err)
	}
	body := captureBody(t, &litellm.Request{
		Model:          "glm-4.5",
		Messages:       []litellm.Message{litellm.UserText("hi")},
		ResponseFormat: format,
	})
	responseFormat := body["response_format"].(map[string]any)
	if responseFormat["type"] != "json_object" {
		t.Fatalf("response_format = %#v", responseFormat)
	}
	messages := body["messages"].([]any)
	content := messageText(messages[0].(map[string]any)["content"])
	if !strings.Contains(content, "Return JSON matching schema answer") {
		t.Fatalf("schema prompt was not injected: %q", content)
	}
}

func messageText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var out strings.Builder
		for _, item := range c {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			text, _ := part["text"].(string)
			out.WriteString(text)
		}
		return out.String()
	default:
		return ""
	}
}

func captureBody(t *testing.T, req *litellm.Request) map[string]any {
	t.Helper()
	var body map[string]any
	p, err := New(compat.Config{
		APIKey:  "key",
		BaseURL: "https://glm.test",
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
