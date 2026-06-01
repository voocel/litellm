package otel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Option configures an OTelHook.
type Option func(*OTelHook)

// WithCaptureContent controls whether prompt and completion text are recorded
// on the span. Enabled by default; disable for privacy or to shrink span size.
func WithCaptureContent(capture bool) Option {
	return func(h *OTelHook) { h.captureContent = capture }
}

// WithSpanAttributes registers a resolver invoked once per call in
// BeforeRequest; the attributes it returns are added to that call's generation
// span. Use it to attach trace-level metadata the gen_ai.* conventions don't
// cover — e.g. a session or user id for backends (such as Langfuse) that group
// generations by those keys. The resolver receives the call's context, so it
// may read values propagated via context or OTel baggage. Returning nil (or an
// empty slice) adds nothing for that call.
func WithSpanAttributes(fn func(ctx context.Context) []attribute.KeyValue) Option {
	return func(h *OTelHook) { h.attrFn = fn }
}

// New returns an OTelHook that emits one generation span per LLM call on the
// given tracer. Register it with litellm.WithHook.
func New(tracer trace.Tracer, opts ...Option) *OTelHook {
	h := &OTelHook{
		tracer:         tracer,
		spans:          make(map[string]*callState),
		captureContent: true,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}
