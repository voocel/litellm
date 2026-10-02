// Package idle fails a response body whose reads wait too long for data, so
// that a stream over a connection that hung ends instead of waiting forever.
// Client.Stream puts the timeout on the context; the HTTP client of every
// provider watches the bodies of responses to requests that carry one.
package idle

import (
	"context"
	"fmt"
	"io"
	"sync/atomic"
	"time"
)

type key struct{}

// WithTimeout returns ctx carrying timeout for the bodies of the responses
// to its requests.
func WithTimeout(ctx context.Context, timeout time.Duration) context.Context {
	return context.WithValue(ctx, key{}, timeout)
}

// Timeout returns the timeout ctx carries, or zero.
func Timeout(ctx context.Context) time.Duration {
	timeout, _ := ctx.Value(key{}).(time.Duration)
	return timeout
}

// Watch returns body, closed once a read has waited timeout for data; that
// read and every later one then fail. Only waiting counts: a consumer slow to
// read is not idle.
func Watch(body io.ReadCloser, timeout time.Duration) io.ReadCloser {
	w := &watched{ReadCloser: body, timeout: timeout}
	w.timer = time.AfterFunc(timeout, w.expire)
	w.timer.Stop()
	return w
}

type watched struct {
	io.ReadCloser
	timeout time.Duration
	timer   *time.Timer
	expired atomic.Bool
}

func (w *watched) expire() {
	w.expired.Store(true)
	w.ReadCloser.Close()
}

func (w *watched) Read(p []byte) (int, error) {
	w.timer.Reset(w.timeout)
	n, err := w.ReadCloser.Read(p)
	w.timer.Stop()
	if w.expired.Load() {
		return n, fmt.Errorf("no data for %v", w.timeout)
	}
	return n, err
}

func (w *watched) Close() error {
	w.timer.Stop()
	return w.ReadCloser.Close()
}
