package anthropic

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestNewRequiresAPIKey(t *testing.T) {
	if _, err := New(Config{}); litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || err.Error() != "anthropic: api key is required" {
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
	want := litellm.Capabilities{MaxTokensRequired: true, ThinkingEffort: true, DisableThinking: true, DeferredTools: true, ProviderOptions: []string{
		"metadata", "output_config", "service_tier", "thinking", "tool_choice", "top_k",
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

// A Client's stream fails once it waits its idle timeout for data, with a
// temporary network error; the vendor's pings count as data.
func TestStreamIdleTimeout(t *testing.T) {
	events := func(w io.Writer, data ...string) {
		for _, d := range data {
			io.WriteString(w, "data: "+d+"\n\n")
		}
	}
	const start = `{"type":"message_start","message":{"model":"claude","usage":{"input_tokens":1}}}`
	stream := func(write func(w io.Writer)) error {
		r, w := io.Pipe()
		go func() {
			write(w)
			w.Close()
		}()
		p, err := New(Config{APIKey: "k", HTTPClient: doFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: r}, nil
		})})
		if err != nil {
			t.Fatal(err)
		}
		client, err := litellm.New(p, litellm.WithStreamIdleTimeout(100*time.Millisecond))
		if err != nil {
			t.Fatal(err)
		}
		s, err := client.Stream(t.Context(), litellm.Request{Model: "claude", MaxTokens: new(64), Messages: []litellm.Message{litellm.UserText("hi")}})
		if err != nil {
			return err
		}
		defer s.Close()
		_, err = litellm.Collect(s)
		return err
	}

	err := stream(func(w io.Writer) {
		events(w, start)
		for range 6 {
			time.Sleep(40 * time.Millisecond)
			events(w, `{"type":"ping"}`)
		}
		events(w,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			`{"type":"message_stop"}`)
	})
	if err != nil {
		t.Fatalf("a stream kept alive by pings failed: %v", err)
	}

	hung := make(chan struct{})
	defer close(hung)
	err = stream(func(w io.Writer) {
		events(w, start)
		<-hung
	})
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeNetwork || !litellm.IsTemporaryError(err) || !strings.Contains(err.Error(), "no data for 100ms") {
		t.Fatalf("hung stream: %v", err)
	}

	// A vendor that never answers fails the same way.
	p, err := New(Config{APIKey: "k", HTTPClient: doFunc(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})})
	if err != nil {
		t.Fatal(err)
	}
	client, err := litellm.New(p, litellm.WithStreamIdleTimeout(100*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Stream(t.Context(), litellm.Request{Model: "claude", MaxTokens: new(64), Messages: []litellm.Message{litellm.UserText("hi")}})
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeNetwork || !litellm.IsTemporaryError(err) || !strings.Contains(err.Error(), "no data for 100ms") {
		t.Fatalf("unanswered call: %v", err)
	}
}

// A named endpoint tags its replies and replay state with its name.
func TestConfigName(t *testing.T) {
	p, err := New(Config{Name: "proxy", APIKey: "k", HTTPClient: doFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(
			`{"content":[{"type":"thinking","thinking":"t","signature":"sig"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		))}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(t.Context(), &litellm.Request{Model: "claude", MaxTokens: new(64), Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != "proxy" || resp.Blocks[0].(litellm.ReasoningBlock).State.Provider != "proxy" {
		t.Fatalf("response = %#v", resp)
	}
}

func TestChatRejectsInvalidRequestBeforeSending(t *testing.T) {
	p := newTestProvider(t, func(*http.Request) (*http.Response, error) {
		t.Fatal("request sent")
		return nil, nil
	})
	_, err := p.Chat(t.Context(), &litellm.Request{Model: "claude", Messages: []litellm.Message{litellm.UserText("hi")}})
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || err.Error() != "anthropic: max_tokens is required by the Messages API" {
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
