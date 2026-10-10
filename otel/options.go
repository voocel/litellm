package otel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Option configures an Observer.
type Option func(*Observer)

// WithCaptureContent controls whether input and output messages are recorded
// on the span. Content capture is disabled by default because messages may
// contain sensitive information. An inline image is recorded as a note of
// its type and size.
func WithCaptureContent(capture bool) Option {
	return func(h *Observer) { h.captureContent = capture }
}

// WithSpanAttributes registers a resolver invoked once per call in
// Start; the attributes it returns are added to that call's generation
// span. Use it to attach trace-level metadata the gen_ai.* conventions don't
// cover — e.g. a session or user id for backends (such as Langfuse) that group
// generations by those keys. The resolver receives the call's context, so it
// may read values propagated via context or OTel baggage. Returning nil (or an
// empty slice) adds nothing for that call.
func WithSpanAttributes(fn func(ctx context.Context) []attribute.KeyValue) Option {
	return func(h *Observer) { h.attrFn = fn }
}

// New returns an Observer that emits one generation span per LLM call on the
// given tracer. Register it with litellm.WithObservers.
func New(tracer trace.Tracer, opts ...Option) *Observer {
	h := &Observer{
		tracer:         tracer,
		captureContent: false,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
