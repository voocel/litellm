package anthropic

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestNewRequiresAPIKey(t *testing.T) {
	if _, err := New(Config{}); !litellm.IsValidationError(err) || err.Error() != "anthropic: api key is required" {
		t.Fatalf("err = %v", err)
	}
	if _, err := New(Config{APIKeyFunc: func(context.Context) (string, error) { return "k", nil }}); err != nil {
		t.Fatalf("APIKeyFunc: %v", err)
	}
}

func TestCapabilities(t *testing.T) {
	p, err := New(Config{APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	want := litellm.Capabilities{Thinking: true, DisableThinking: true, ThinkingEffort: true, ThinkingBudget: true, MaxTokensRequired: true, ProviderOptions: []string{
		"container", "context_management", "mcp_servers", "metadata", "output_config", "service_tier", "tool_choice", "tools", "top_k",
	}}
	if got := p.Capabilities(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Capabilities = %+v", got)
	}
}

func TestChatSendsHeaders(t *testing.T) {
	p, err := New(Config{
		APIKeyFunc: func(context.Context) (string, error) { return "key-from-func", nil },
		BaseURL:    "https://example.test/",
		UserAgent:  "my-client/1.0",
		Headers:    map[string]string{"X-Custom": "v", "anthropic-beta": "beta-name"},
		HTTPClient: doFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.String() != "https://example.test/v1/messages" {
				t.Errorf("url = %s", req.URL)
			}
			for key, want := range map[string]string{
				"x-api-key": "key-from-func", "anthropic-version": "2023-06-01", "anthropic-beta": "beta-name",
				"User-Agent": "my-client/1.0", "X-Custom": "v", "Content-Type": "application/json", "Accept": "",
			} {
				if got := req.Header.Get(key); got != want {
					t.Errorf("%s = %q, want %q", key, got, want)
				}
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(
				`{"model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			))}, nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(t.Context(), &litellm.Request{Model: "claude", MaxTokens: new(64), Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil || resp.Text() != "ok" {
		t.Fatalf("Chat = %+v, %v", resp, err)
	}
}

func TestChatRejectsInvalidRequestBeforeSending(t *testing.T) {
	p := newTestProvider(t, func(*http.Request) (*http.Response, error) {
		t.Fatal("request sent")
		return nil, nil
	})
	_, err := p.Chat(t.Context(), &litellm.Request{Model: "claude", Messages: []litellm.Message{litellm.UserText("hi")}})
	if !litellm.IsValidationError(err) || err.Error() != "anthropic: max_tokens is required by the Messages API" {
		t.Fatalf("err = %v", err)
	}
}

type doFunc func(*http.Request) (*http.Response, error)

func (f doFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

func newTestProvider(t *testing.T, do doFunc) *Provider {
	t.Helper()
	p, err := New(Config{APIKey: "k", BaseURL: "https://example.test", HTTPClient: do})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func fixtureResponse(t *testing.T, name string) *http.Response {
	t.Helper()
	body := testgolden.ReadFixture(t, "../../testdata/anthropic/"+name)
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(string(body)))}
}
