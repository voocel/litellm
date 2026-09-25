// Package compattest runs openaicompat providers against canned HTTP
// responses in tests.
package compattest

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

// NewFunc builds a provider, like a vendor package's New.
type NewFunc func(openaicompat.Config) (*openaicompat.Provider, error)

// Spec returns a NewFunc for spec.
func Spec(spec openaicompat.Spec) NewFunc {
	return func(cfg openaicompat.Config) (*openaicompat.Provider, error) {
		return openaicompat.New(cfg, spec)
	}
}

// Doer adapts a function to litellm.HTTPClient.
type Doer func(*http.Request) (*http.Response, error)

func (f Doer) Do(req *http.Request) (*http.Response, error) { return f(req) }

// Response returns a 200 response with body.
func Response(body string) *http.Response {
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

// SSE joins payloads into an event stream ending with [DONE].
func SSE(data ...string) string {
	var out strings.Builder
	for _, d := range append(data, "[DONE]") {
		out.WriteString("data: " + d + "\n\n")
	}
	return out.String()
}

// Provider builds a provider with an API key, a test base URL and client.
func Provider(t testing.TB, newFn NewFunc, client litellm.HTTPClient) *openaicompat.Provider {
	t.Helper()
	p, err := newFn(openaicompat.Config{APIKey: "key", BaseURL: "https://api.test/v1", HTTPClient: client})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// Request is a minimal valid request.
func Request() *litellm.Request {
	return &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}}
}

// Body sends req, as a stream when stream is set, and returns the JSON body.
func Body(t testing.TB, newFn NewFunc, req *litellm.Request, stream bool) map[string]any {
	t.Helper()
	var body map[string]any
	p := Provider(t, newFn, Doer(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		if stream {
			return Response(SSE()), nil
		}
		return Response(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`), nil
	}))
	var err error
	if stream {
		var s litellm.Stream
		if s, err = p.Stream(context.Background(), req); err == nil {
			s.Close()
		}
	} else {
		_, err = p.Chat(context.Background(), req)
	}
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	return body
}

// Err returns the error for req, which must fail before a request is sent.
func Err(t testing.TB, newFn NewFunc, req *litellm.Request) error {
	t.Helper()
	p := Provider(t, newFn, Doer(func(*http.Request) (*http.Response, error) {
		t.Fatal("request was sent")
		return nil, nil
	}))
	_, err := p.Chat(context.Background(), req)
	if err == nil {
		t.Fatal("Chat succeeded")
	}
	return err
}

// Chat returns the response to a canned JSON body. It checks that Raw is the
// body and clears it, so the result compares with a collected stream.
func Chat(t testing.TB, newFn NewFunc, body string) (*litellm.Response, error) {
	t.Helper()
	p := Provider(t, newFn, Doer(func(*http.Request) (*http.Response, error) { return Response(body), nil }))
	resp, err := p.Chat(context.Background(), Request())
	if resp != nil {
		if string(resp.Raw) != body {
			t.Fatalf("Raw = %s, want the response body", resp.Raw)
		}
		resp.Raw = nil
	}
	return resp, err
}

// Stream opens a stream over a canned event-stream body.
func Stream(t testing.TB, newFn NewFunc, sse string) litellm.Stream {
	t.Helper()
	p := Provider(t, newFn, Doer(func(*http.Request) (*http.Response, error) { return Response(sse), nil }))
	s, err := p.Stream(context.Background(), Request())
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// Events reads a canned stream to its end. The error is the one that ended
// it, or nil at EOF.
func Events(t testing.TB, newFn NewFunc, sse string) ([]litellm.Event, error) {
	t.Helper()
	s := Stream(t, newFn, sse)
	var events []litellm.Event
	for {
		event, err := s.Next()
		if errors.Is(err, io.EOF) {
			return events, nil
		}
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
}

// Collect aggregates a canned stream.
func Collect(t testing.TB, newFn NewFunc, sse string) (*litellm.Response, error) {
	t.Helper()
	return litellm.Collect(Stream(t, newFn, sse))
}

// Options encodes ProviderOptions.
func Options(t testing.TB, values map[string]any) litellm.ProviderOptions {
	t.Helper()
	o, err := litellm.NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// AssertJSON compares got, a value or JSON bytes, with the JSON text want.
func AssertJSON(t testing.TB, got any, want string) {
	t.Helper()
	if g, w := canonical(t, got), canonical(t, []byte(want)); g != w {
		t.Fatalf("JSON mismatch\n got: %s\nwant: %s", g, w)
	}
}

// AssertFields compares the top-level fields of body named in want, a JSON
// object. A null in want means the field is absent.
func AssertFields(t testing.TB, body map[string]any, want string) {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(want), &fields); err != nil {
		t.Fatalf("decode want: %v", err)
	}
	for key, value := range fields {
		if g, w := canonical(t, body[key]), canonical(t, []byte(value)); g != w {
			t.Errorf("%s = %s, want %s", key, g, w)
		}
	}
}

func canonical(t testing.TB, v any) string {
	t.Helper()
	data, ok := v.([]byte)
	if !ok {
		var err error
		if data, err = json.Marshal(v); err != nil {
			t.Fatalf("marshal: %v", err)
		}
	}
	var decoded any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("decode %s: %v", data, err)
	}
	out, _ := json.Marshal(decoded)
	return string(out)
}
