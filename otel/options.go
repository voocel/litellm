package otel

import "go.opentelemetry.io/otel/trace"

// Option configures an OTelHook.
type Option func(*OTelHook)

// WithCaptureContent controls whether prompt and completion text are recorded
// on the span. Enabled by default; disable for privacy or to shrink span size.
func WithCaptureContent(capture bool) Option {
	return func(h *OTelHook) { h.captureContent = capture }
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
