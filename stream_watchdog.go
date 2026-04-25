package litellm

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/voocel/litellm/providers"
)

// ErrStreamIdle is the sentinel cause for stream idle-timeout errors.
// Use errors.Is(err, ErrStreamIdle) or IsStreamIdleError(err) to detect.
var ErrStreamIdle = errors.New("stream idle timeout: no chunks received")

// IsStreamIdleError reports whether err originated from a stream idle-timeout abort.
func IsStreamIdleError(err error) bool {
	return errors.Is(err, ErrStreamIdle)
}

// streamIdleWatchdog wraps a StreamReader to enforce a per-chunk idle timeout.
// If no chunk arrives within the configured timeout the underlying request
// context is cancelled and Next returns a timeout LiteLLMError whose cause is
// ErrStreamIdle.
//
// Background: an http.Client.Timeout (RequestTimeout) covers the entire
// request, including streaming body reads. When an upstream proxy or provider
// stops emitting tokens mid-stream without closing the TCP connection, body
// reads block until that total deadline fires (often 10 minutes), masking the
// stall. The watchdog is the per-chunk inactivity check that makes such stalls
// surface in seconds. Modeled after Claude Code's STREAM_IDLE_TIMEOUT_MS
// (services/api/claude.ts).
type streamIdleWatchdog struct {
	inner    providers.StreamReader
	cancel   context.CancelFunc
	timeout  time.Duration
	provider string

	timer   *time.Timer
	aborted atomic.Bool
}

// newStreamIdleWatchdog wraps inner with idle-timeout enforcement.
// Returns inner unchanged when timeout <= 0 or inner is nil so callers can
// safely use it unconditionally.
func newStreamIdleWatchdog(inner providers.StreamReader, cancel context.CancelFunc, timeout time.Duration, provider string) providers.StreamReader {
	if inner == nil || timeout <= 0 {
		return inner
	}
	w := &streamIdleWatchdog{
		inner:    inner,
		cancel:   cancel,
		timeout:  timeout,
		provider: provider,
	}
	w.timer = time.AfterFunc(timeout, w.fire)
	return w
}

func (w *streamIdleWatchdog) fire() {
	if !w.aborted.CompareAndSwap(false, true) {
		return
	}
	if w.cancel != nil {
		w.cancel()
	}
}

func (w *streamIdleWatchdog) Next() (*StreamChunk, error) {
	chunk, err := w.inner.Next()
	if err != nil {
		if w.aborted.Load() {
			return nil, w.idleError()
		}
		return nil, err
	}
	if !w.aborted.Load() {
		w.timer.Reset(w.timeout)
	}
	return chunk, nil
}

func (w *streamIdleWatchdog) Close() error {
	if w.timer != nil {
		w.timer.Stop()
	}
	err := w.inner.Close()
	if w.cancel != nil {
		w.cancel()
	}
	return err
}

func (w *streamIdleWatchdog) idleError() error {
	return &LiteLLMError{
		Type:      ErrorTypeTimeout,
		Provider:  w.provider,
		Message:   fmt.Sprintf("stream idle timeout: no chunks received for %s", w.timeout),
		Cause:     ErrStreamIdle,
		Retryable: true,
	}
}

// resilienceConfigProvider is the optional interface implemented by providers
// that expose their effective ResilienceConfig. BaseProvider satisfies it, so
// all builtin providers do too.
type resilienceConfigProvider interface {
	ResilienceConfig() providers.ResilienceConfig
}

// resolveStreamWatchdog returns a derived context for the streaming call along
// with the cancel + timeout to feed into the watchdog wrapper. When the
// provider does not expose a resilience config or has the watchdog disabled,
// it returns the original context with a nil cancel (no wrapping).
func (c *Client) resolveStreamWatchdog(ctx context.Context) (context.Context, context.CancelFunc, time.Duration) {
	rcp, ok := c.provider.(resilienceConfigProvider)
	if !ok {
		return ctx, nil, 0
	}
	timeout := rcp.ResilienceConfig().StreamIdleTimeout
	if timeout <= 0 {
		return ctx, nil, 0
	}
	streamCtx, cancel := context.WithCancel(ctx)
	return streamCtx, cancel, timeout
}

// attachStreamWatchdog finalises the watchdog binding for a stream call.
// It wraps stream when invocation succeeded with a non-nil reader, and
// otherwise releases the derived stream context to avoid leaks. Safe to call
// with a nil cancel (no-op).
func (c *Client) attachStreamWatchdog(stream providers.StreamReader, invokeErr error, cancel context.CancelFunc, timeout time.Duration) providers.StreamReader {
	if cancel == nil {
		return stream
	}
	if invokeErr != nil || stream == nil {
		cancel()
		return stream
	}
	return newStreamIdleWatchdog(stream, cancel, timeout, c.provider.Name())
}
