package litellm

import (
	"context"
	"errors"
	"io"
	"time"
)

// CallInfo describes the caller's request before validation. Request is a
// snapshot shared by all observers: read it, never modify it.
type CallInfo struct {
	Provider  string
	Streaming bool
	StartedAt time.Time
	Request   *Request
}

// CallStatus is how an invocation ended. CallClosed means the caller closed a
// stream before it finished.
type CallStatus string

const (
	CallCompleted CallStatus = "completed"
	CallFailed    CallStatus = "failed"
	CallCanceled  CallStatus = "canceled"
	CallClosed    CallStatus = "closed"
)

// CallResult describes SDK execution, not errors in application consumer callbacks.
// Response may be partial on failure or early Close. A later resource cleanup
// error does not revise an already completed invocation; Close returns that error.
type CallResult struct {
	Status   CallStatus
	Response *Response
	Err      error
	Duration time.Duration
}

// Observer starts one observation per Chat/Stream invocation, including local
// validation failures. Its returned non-nil context reaches subsequent observers
// and the provider; derive it from the input to preserve cancellation and values.
// Returning a nil CallObserver opts out of observing this call.
// Callbacks are synchronous; the SDK does not recover observer panics.
type Observer interface {
	Start(context.Context, CallInfo) (context.Context, CallObserver)
}

// ObserverFunc adapts a function to Observer.
type ObserverFunc func(context.Context, CallInfo) (context.Context, CallObserver)

// Start calls f.
func (f ObserverFunc) Start(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
	return f(ctx, info)
}

// CallObserver belongs to one invocation. Events and the Response are snapshots
// shared by all observers and isolated from the caller: read, never modify.
// OnEvent receives stream events as they arrive, and Chat warnings as
// WarningEvent. End runs exactly once; observers end in
// reverse registration order so nested observation scopes unwind correctly.
// Streams must be consumed to termination or explicitly closed; cancellation
// alone does not run callbacks in a background goroutine.
type CallObserver interface {
	OnEvent(Event)
	End(CallResult)
}

// WithObservers adds observers, skipping nil ones.
func WithObservers(observers ...Observer) ClientOption {
	return func(c *Client) error {
		for _, observer := range observers {
			if observer != nil {
				c.observers = append(c.observers, observer)
			}
		}
		return nil
	}
}

type callObservation struct {
	started   time.Time
	observers []CallObserver
	ended     bool
}

func (c *Client) startCall(ctx context.Context, req Request, streaming bool) (context.Context, *callObservation) {
	call := &callObservation{started: time.Now()}
	if len(c.observers) == 0 {
		return ctx, call
	}
	info := CallInfo{Provider: c.ProviderName(), Streaming: streaming, StartedAt: call.started, Request: cloneRequest(req)}
	for _, observer := range c.observers {
		var active CallObserver
		ctx, active = observer.Start(ctx, info)
		if active != nil {
			call.observers = append(call.observers, active)
		}
	}
	return ctx, call
}

// Observers share one snapshot per event, isolated from the caller's copy.
func (c *callObservation) event(e Event) {
	if len(c.observers) == 0 {
		return
	}
	e = cloneEvent(e)
	for _, observer := range c.observers {
		observer.OnEvent(e)
	}
}
func (c *callObservation) end(status CallStatus, resp *Response, err error) {
	if c.ended {
		return
	}
	c.ended = true
	duration := time.Since(c.started)
	result := CallResult{Status: status, Response: cloneResponse(resp), Err: err, Duration: duration}
	for i := len(c.observers) - 1; i >= 0; i-- {
		c.observers[i].End(result)
	}
}
func callStatus(err error) CallStatus {
	// Timeout cleanup may also return context.Canceled; retain the timeout as
	// the outcome rather than interpreting its cancellation as a caller action.
	if errors.Is(err, context.DeadlineExceeded) || IsTimeoutError(err) {
		return CallFailed
	}
	if errors.Is(err, context.Canceled) {
		return CallCanceled
	}
	if err != nil {
		return CallFailed
	}
	return CallCompleted
}

type observedStream struct {
	ctx      context.Context
	cancel   context.CancelFunc
	call     *callObservation
	inner    Stream
	closed   bool
	closeErr error
}

func (s *observedStream) eventCollector() *collector { return streamCollector(s.inner) }
func (s *observedStream) finish(status CallStatus, err error) {
	if s.call.ended {
		return
	}
	var resp *Response
	if len(s.call.observers) > 0 {
		resp = s.eventCollector().Response()
	}
	s.call.end(status, resp, err)
	s.cancel()
}
func (s *observedStream) Next() (Event, error) {
	if s.call.ended {
		return nil, io.EOF
	}
	event, err := s.inner.Next()
	if err != nil {
		s.finish(callStatus(err), err)
		return nil, err
	}
	s.call.event(event)
	if _, ok := event.(DoneEvent); ok {
		s.finish(CallCompleted, nil)
	}
	return event, nil
}
func (s *observedStream) Close() error {
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	// Read cancellation before closing inner: our own cleanup also cancels ctx.
	ctxErr := s.ctx.Err()
	s.closeErr = s.inner.Close()
	if !s.call.ended {
		switch {
		case s.closeErr != nil:
			s.finish(callStatus(s.closeErr), s.closeErr)
		case ctxErr != nil:
			s.finish(callStatus(ctxErr), ctxErr)
		default:
			s.finish(CallClosed, nil)
		}
	}
	s.cancel()
	return s.closeErr
}
