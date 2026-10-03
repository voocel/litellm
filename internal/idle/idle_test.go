package idle

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func respond(body io.ReadCloser) func(*http.Request) (*http.Response, error) {
	return func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Body: body}, nil
	}
}

func TestReadWaitingTooLongFails(t *testing.T) {
	r, w := io.Pipe()
	defer w.Close()
	resp, err := Do(respond(r), httptestRequest(t), 20*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		w.Write([]byte("a"))
		time.Sleep(10 * time.Millisecond)
		w.Write([]byte("b")) // within the timeout of the read waiting for it
	}()
	buf := make([]byte, 1)
	for _, want := range "ab" {
		if n, err := resp.Body.Read(buf); err != nil || n != 1 || rune(buf[0]) != want {
			t.Fatalf("read %q, %v", buf[:n], err)
		}
	}
	if _, err := resp.Body.Read(buf); err == nil || !strings.Contains(err.Error(), "no data for 20ms") {
		t.Fatalf("idle read: %v", err)
	}
	if _, err := resp.Body.Read(buf); err == nil {
		t.Fatal("a read after the timeout succeeded")
	}
}

// Only a read waiting counts: data a consumer is slow to read is no sign of
// a connection that hung.
func TestSlowConsumerIsNotIdle(t *testing.T) {
	resp, err := Do(respond(io.NopCloser(strings.NewReader("data"))), httptestRequest(t), 10*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(30 * time.Millisecond)
	if data, err := io.ReadAll(resp.Body); err != nil || string(data) != "data" {
		t.Fatalf("read %q, %v", data, err)
	}
}

// A response that never comes fails the call as idle, not as the caller's
// cancellation, and ends the request.
func TestResponseWaitingTooLongFails(t *testing.T) {
	ended := make(chan error, 1)
	send := func(req *http.Request) (*http.Response, error) {
		<-req.Context().Done()
		ended <- req.Context().Err()
		return nil, req.Context().Err()
	}
	_, err := Do(send, httptestRequest(t), 20*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "no data for 20ms") || errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if err := <-ended; !errors.Is(err, context.Canceled) {
		t.Fatalf("request ended with %v", err)
	}
}

// The request lives as long as its body.
func TestClosingTheBodyEndsTheRequest(t *testing.T) {
	var ctx context.Context
	send := func(req *http.Request) (*http.Response, error) {
		ctx = req.Context()
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("data"))}, nil
	}
	resp, err := Do(send, httptestRequest(t), time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if ctx.Err() != nil {
		t.Fatal("the request ended before its body was closed")
	}
	resp.Body.Close()
	if ctx.Err() == nil {
		t.Fatal("the request outlived its body")
	}
}

func httptestRequest(t *testing.T) *http.Request {
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, "https://example.test", nil)
	if err != nil {
		t.Fatal(err)
	}
	return req
}
