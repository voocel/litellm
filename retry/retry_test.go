package retry

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestTransportRetriesCompleteRetryableResponses(t *testing.T) {
	var attempts int
	transport := newTransport(roundTripFunc(func(req *http.Request) (*http.Response, error) {
		attempts++
		if attempts == 1 {
			return response(http.StatusTooManyRequests, "slow down"), nil
		}
		body, err := io.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		if string(body) != `{"ok":true}` {
			t.Fatalf("body = %q", body)
		}
		return response(http.StatusOK, "ok"), nil
	}), &Policy{MaxAttempts: 2, InitialDelay: time.Nanosecond})

	req, err := http.NewRequest(http.MethodPost, "https://example.test", bytes.NewReader([]byte(`{"ok":true}`)))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	defer resp.Body.Close()
	if attempts != 2 || resp.StatusCode != http.StatusOK {
		t.Fatalf("attempts/status = %d/%d", attempts, resp.StatusCode)
	}
}

// Only failures the provider reports as temporary are retried, and the body
// read to classify one reaches the provider intact.
func TestTransportRetriesTemporaryFailuresOnly(t *testing.T) {
	quota := `{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}`
	for _, tc := range []struct {
		name       string
		resp       func() *http.Response
		wantStatus int
		wantBody   string
	}{
		{"exhausted quota", func() *http.Response { return response(http.StatusTooManyRequests, quota) }, http.StatusTooManyRequests, quota},
		{"bad request", func() *http.Response { return response(http.StatusBadRequest, "bad") }, http.StatusBadRequest, "bad"},
		{"wait beyond MaxRetryAfter", func() *http.Response {
			resp := response(http.StatusTooManyRequests, "slow down")
			resp.Header.Set("Retry-After", "61")
			return resp
		}, http.StatusTooManyRequests, "slow down"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var attempts int
			transport := newTransport(roundTripFunc(func(*http.Request) (*http.Response, error) {
				attempts++
				return tc.resp(), nil
			}), DefaultPolicy())
			req, err := http.NewRequest(http.MethodGet, "https://example.test", nil)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			if attempts != 1 || resp.StatusCode != tc.wantStatus || string(body) != tc.wantBody {
				t.Fatalf("attempts=%d status=%d body=%q", attempts, resp.StatusCode, body)
			}
		})
	}
}

func TestTransportDoesNotRetryNetworkErrors(t *testing.T) {
	boom := errors.New("boom")
	var attempts int
	transport := newTransport(roundTripFunc(func(req *http.Request) (*http.Response, error) {
		attempts++
		return nil, boom
	}), &Policy{MaxAttempts: 3, InitialDelay: time.Nanosecond})

	req, err := http.NewRequest(http.MethodGet, "https://example.test", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	_, err = transport.RoundTrip(req)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want boom", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func TestTransportRetryAfterRespectsContext(t *testing.T) {
	var attempts int
	transport := newTransport(roundTripFunc(func(req *http.Request) (*http.Response, error) {
		attempts++
		resp := response(http.StatusTooManyRequests, "slow down")
		resp.Header.Set("Retry-After", "30")
		return resp, nil
	}), &Policy{MaxAttempts: 2, InitialDelay: time.Hour, RespectRetryAfter: true})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	_, err = transport.RoundTrip(req)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context deadline", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func TestTransportReturnsResponseForNonReplayableBody(t *testing.T) {
	var attempts int
	transport := newTransport(roundTripFunc(func(req *http.Request) (*http.Response, error) {
		attempts++
		return response(http.StatusServiceUnavailable, "retry"), nil
	}), &Policy{MaxAttempts: 2, InitialDelay: time.Nanosecond})

	req, err := http.NewRequest(http.MethodPost, "https://example.test", io.NopCloser(strings.NewReader("body")))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("resp = %v, err = %v; want the original 503", resp, err)
	}
	resp.Body.Close()
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func response(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestTransportDoesNotReplayInterruptedSuccessfulResponse(t *testing.T) {
	boom := errors.New("stream interrupted")
	attempts := 0
	transport := newTransport(roundTripFunc(func(req *http.Request) (*http.Response, error) {
		attempts++
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: &failingBody{err: boom}}, nil
	}), &Policy{MaxAttempts: 3, InitialDelay: time.Nanosecond})
	req, err := http.NewRequest(http.MethodPost, "https://example.test", strings.NewReader("request"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if _, err := io.ReadAll(resp.Body); !errors.Is(err, boom) {
		t.Fatalf("read error = %v", err)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

type failingBody struct{ err error }

func (b *failingBody) Read([]byte) (int, error) { return 0, b.err }
func (b *failingBody) Close() error             { return nil }
