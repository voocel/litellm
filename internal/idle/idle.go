// Package idle fails a call over a connection that hung: one that waits too
// long for its response or, once that arrives, for data from its body.
// Client.Stream puts the timeout on the context; the HTTP client of every
// provider sends requests that carry one through Do.
package idle

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"time"
)

type key struct{}

// WithTimeout returns ctx carrying timeout for the calls made with it.
func WithTimeout(ctx context.Context, timeout time.Duration) context.Context {
	return context.WithValue(ctx, key{}, timeout)
}

// Timeout returns the timeout ctx carries, or zero.
func Timeout(ctx context.Context) time.Duration {
	timeout, _ := ctx.Value(key{}).(time.Duration)
	return timeout
}

// Do sends req with send and returns the response, failing the call once it
// waits timeout for data: for the response, the waits of any retries send
// makes included, or then for a read of its body. Only waiting counts: a
// consumer slow to read is not idle.
func Do(send func(*http.Request) (*http.Response, error), req *http.Request, timeout time.Duration) (*http.Response, error) {
	ctx, cancel := context.WithCancel(req.Context())
	timer := time.AfterFunc(timeout, cancel)
	resp, err := send(req.WithContext(ctx))
	if !timer.Stop() {
		if err == nil {
			resp.Body.Close()
		}
		return nil, idleError(timeout)
	}
	if err != nil {
		cancel()
		return nil, err
	}
	body := &watched{ReadCloser: resp.Body, timeout: timeout, cancel: cancel}
	body.timer = time.AfterFunc(timeout, body.expire)
	body.timer.Stop()
	resp.Body = body
	return resp, nil
}

// idleError does not wrap the cancellation that ended the wait, which would
// read as the caller's.
func idleError(timeout time.Duration) error {
	return fmt.Errorf("no data for %v", timeout)
}

// watched is a body closed once a read has waited timeout for data; that
// read and every later one then fail.
type watched struct {
	io.ReadCloser
	timeout time.Duration
	timer   *time.Timer
	expired atomic.Bool
	cancel  context.CancelFunc
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
		return n, idleError(w.timeout)
	}
	return n, err
}

func (w *watched) Close() error {
	w.timer.Stop()
	defer w.cancel()
	return w.ReadCloser.Close()
}
