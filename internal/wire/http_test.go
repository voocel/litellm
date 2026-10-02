package wire

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/voocel/litellm"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

type trackingBody struct {
	io.Reader
	closed bool
}

func (b *trackingBody) Close() error { b.closed = true; return nil }

func TestDoMapsHTTPErrorWithRetryAfter(t *testing.T) {
	body := &trackingBody{Reader: strings.NewReader(`{"error":{"code":"rate_limit","message":"slow down"}}`)}
	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": {"7"}}, Body: body}, nil
	})
	req, _ := http.NewRequest(http.MethodPost, "https://example.test", nil)
	_, err := Do(client, req, "test", "request")
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit || litellm.RetryAfter(err) != 7*time.Second {
		t.Fatalf("err = %v, retry after = %v", err, litellm.RetryAfter(err))
	}
	if !body.closed {
		t.Fatal("error response body not closed")
	}
}

// Google APIs suggest the wait in the error body; a Retry-After header wins.
func TestHTTPErrorReadsGoogleRetryDelay(t *testing.T) {
	body := `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"quota","details":[` +
		`{"@type":"type.googleapis.com/google.rpc.QuotaFailure"},` +
		`{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"18.5s"}]}}`
	if got := HTTPError("gemini", http.StatusTooManyRequests, nil, body).RetryAfter; got != 18500*time.Millisecond {
		t.Fatalf("retry after = %v, want 18.5s", got)
	}
	if got := HTTPError("gemini", http.StatusTooManyRequests, http.Header{"Retry-After": {"7"}}, body).RetryAfter; got != 7*time.Second {
		t.Fatalf("retry after = %v, want the header's 7s", got)
	}
}

func TestDoMapsTransportFailure(t *testing.T) {
	client := doerFunc(func(*http.Request) (*http.Response, error) { return nil, errors.New("reset") })
	req, _ := http.NewRequest(http.MethodPost, "https://example.test", nil)
	_, err := Do(client, req, "test", "stream request")
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeNetwork || !strings.Contains(err.Error(), "stream request failed") {
		t.Fatalf("err = %v", err)
	}
}

func TestDoReturnsSuccessfulResponseOpen(t *testing.T) {
	body := &trackingBody{Reader: strings.NewReader("ok")}
	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: body}, nil
	})
	req, _ := http.NewRequest(http.MethodPost, "https://example.test", nil)
	resp, err := Do(client, req, "test", "request")
	if err != nil || resp == nil || body.closed {
		t.Fatalf("resp = %v, err = %v, closed = %v", resp, err, body.closed)
	}
}

func TestDoRejectsHTMLPage(t *testing.T) {
	body := &trackingBody{Reader: strings.NewReader("\n  <!doctype html><html></html>")}
	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/html; charset=utf-8"}}, Body: body}, nil
	})
	req, _ := http.NewRequest(http.MethodPost, "https://example.test/chat/completions?key=secret", nil)
	_, err := Do(client, req, "test", "request")
	want := "test: https://example.test/chat/completions returned an HTML page instead of an API response; check BaseURL"
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeProvider || err.Error() != want {
		t.Fatalf("err = %v", err)
	}
	if !body.closed {
		t.Fatal("HTML response body not closed")
	}
}

func TestDoKeepsJSONLabeledAsHTML(t *testing.T) {
	body := &trackingBody{Reader: strings.NewReader(` {"ok":true}`)}
	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/html"}}, Body: body}, nil
	})
	req, _ := http.NewRequest(http.MethodPost, "https://example.test", nil)
	resp, err := Do(client, req, "test", "request")
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(resp.Body)
	if string(data) != ` {"ok":true}` {
		t.Fatalf("body = %q", data)
	}
	resp.Body.Close()
	if !body.closed {
		t.Fatal("underlying body not closed")
	}
}

func TestStreamErrorClassifiesErrorFields(t *testing.T) {
	tests := map[string]struct {
		raw  string
		want litellm.ErrorType
	}{
		"openrouter status":   {`{"code":429,"message":"Rate limited","metadata":{"error_type":"rate_limit_exceeded"}}`, litellm.ErrorTypeRateLimit},
		"openrouter overflow": {`{"code":400,"message":"This endpoint's maximum context length is 8192 tokens"}`, litellm.ErrorTypeContextOverflow},
		"openai type":         {`{"message":"The server had an error","type":"server_error","param":null,"code":null}`, litellm.ErrorTypeProvider},
		"string code":         {`{"message":"slow down","code":"rate_limit_exceeded"}`, litellm.ErrorTypeRateLimit},
		"string error":        {`"upstream failed"`, litellm.ErrorTypeProvider},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			err := ErrorField("p", json.RawMessage(tt.raw))
			if litellm.ErrorTypeOf(err) != tt.want {
				t.Fatalf("err = %v", err)
			}
		})
	}
	for _, raw := range []string{"", "null"} {
		if err := ErrorField("p", json.RawMessage(raw)); err != nil {
			t.Fatalf("ErrorField(%q) = %v, want nil", raw, err)
		}
	}
}
