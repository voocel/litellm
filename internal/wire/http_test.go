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
	if !litellm.IsRateLimitError(err) || litellm.RetryAfter(err) != 7*time.Second {
		t.Fatalf("err = %v, retry after = %v", err, litellm.RetryAfter(err))
	}
	if !body.closed {
		t.Fatal("error response body not closed")
	}
}

func TestDoMapsTransportFailure(t *testing.T) {
	client := doerFunc(func(*http.Request) (*http.Response, error) { return nil, errors.New("reset") })
	req, _ := http.NewRequest(http.MethodPost, "https://example.test", nil)
	_, err := Do(client, req, "test", "stream request")
	if !litellm.IsNetworkError(err) || !strings.Contains(err.Error(), "stream request failed") {
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

func TestStreamErrorClassifiesErrorFields(t *testing.T) {
	tests := map[string]struct {
		raw string
		is  func(error) bool
	}{
		"openrouter status":   {`{"code":429,"message":"Rate limited","metadata":{"error_type":"rate_limit_exceeded"}}`, litellm.IsRateLimitError},
		"openrouter overflow": {`{"code":400,"message":"This endpoint's maximum context length is 8192 tokens"}`, litellm.IsContextOverflowError},
		"openai type":         {`{"message":"The server had an error","type":"server_error","param":null,"code":null}`, litellm.IsProviderError},
		"string code":         {`{"message":"slow down","code":"rate_limit_exceeded"}`, litellm.IsRateLimitError},
		"string error":        {`"upstream failed"`, litellm.IsProviderError},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			err := ErrorField("p", json.RawMessage(tt.raw))
			if !tt.is(err) {
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
