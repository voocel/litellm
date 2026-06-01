// Package otel adapts litellm's execution Hook into OpenTelemetry generation
// spans. Spans carry gen_ai.* semantic-convention attributes, so any OTLP
// backend (Langfuse, Phoenix, Jaeger, …) renders each LLM call as a generation
// — without litellm itself depending on OpenTelemetry.
package otel

import (
	"context"
	"encoding/json"
	"strings"
	"sync"

	"github.com/voocel/litellm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// callState tracks an in-flight call between the BeforeRequest /
// OnStreamChunk / AfterResponse callbacks, keyed by litellm CallMeta.CallID.
// content accumulates streamed output (non-streaming reads it from Response).
type callState struct {
	span    trace.Span
	content strings.Builder
}

// OTelHook implements litellm.Hook, emitting one OpenTelemetry span per LLM
// call. litellm invokes hooks synchronously and without panic isolation, so
// every method recovers internally — observability must never break the call.
type OTelHook struct {
	tracer         trace.Tracer
	captureContent bool
	// attrFn, when set, is invoked once per call in BeforeRequest; the
	// attributes it returns are added to that call's generation span. It lets
	// the embedder attach trace-level metadata (session id, user id, …) that
	// the gen_ai.* conventions don't cover, reading from the call's context.
	attrFn func(ctx context.Context) []attribute.KeyValue

	mu    sync.Mutex
	spans map[string]*callState
}

var _ litellm.Hook = (*OTelHook)(nil)

// BeforeRequest opens the span and records request-side attributes.
func (h *OTelHook) BeforeRequest(ctx context.Context, meta litellm.CallMeta, req *litellm.Request) {
	defer recoverHook()
	_, span := h.tracer.Start(ctx, meta.Operation+" "+meta.Model)
	span.SetAttributes(
		attribute.String(attrSystem, meta.Provider),
		attribute.String(attrRequestModel, meta.Model),
	)
	if h.attrFn != nil {
		if extra := h.attrFn(ctx); len(extra) > 0 {
			span.SetAttributes(extra...)
		}
	}
	if h.captureContent && req != nil && len(req.Messages) > 0 {
		if data, err := json.Marshal(req.Messages); err == nil {
			span.SetAttributes(attribute.String(attrPrompt, string(data)))
		}
	}
	h.mu.Lock()
	h.spans[meta.CallID] = &callState{span: span}
	h.mu.Unlock()
}

// AfterResponse finishes non-streaming spans. For streaming calls it fires when
// the stream is established (resp == nil) — the span is finished in
// OnStreamChunk on the final chunk; only a stream-setup error is handled here.
func (h *OTelHook) AfterResponse(ctx context.Context, meta litellm.CallMeta, resp *litellm.Response, err error) {
	defer recoverHook()
	if meta.Streaming && err == nil {
		return
	}
	st := h.take(meta.CallID)
	if st == nil {
		return
	}
	if err != nil {
		st.span.RecordError(err)
		st.span.SetStatus(codes.Error, err.Error())
		st.span.End()
		return
	}
	if resp != nil {
		stampResponse(st.span, resp.Model, resp.FinishReason, &resp.Usage)
		if h.captureContent && resp.Content != "" {
			st.span.SetAttributes(attribute.String(attrCompletion, resp.Content))
		}
	}
	st.span.End()
}

// OnStreamChunk accumulates streamed output and finishes the span on the final
// chunk, where the aggregated usage becomes available.
func (h *OTelHook) OnStreamChunk(ctx context.Context, meta litellm.CallMeta, chunk *litellm.StreamChunk) {
	defer recoverHook()
	if chunk == nil {
		return
	}
	h.mu.Lock()
	st := h.spans[meta.CallID]
	h.mu.Unlock()
	if st == nil {
		return
	}
	if h.captureContent && chunk.Content != "" {
		st.content.WriteString(chunk.Content)
	}
	if !chunk.Done {
		return
	}
	stampResponse(st.span, chunk.Model, chunk.FinishReason, chunk.Usage)
	if h.captureContent && st.content.Len() > 0 {
		st.span.SetAttributes(attribute.String(attrCompletion, st.content.String()))
	}
	st.span.End()
	h.mu.Lock()
	delete(h.spans, meta.CallID)
	h.mu.Unlock()
}

// OnStreamEnd finalizes a streaming span that did not complete through a final
// Done chunk — the stream aborted (provider error, context cancel; err != nil)
// or the caller closed it early (err == nil). When the stream completed normally
// the span was already ended in OnStreamChunk and removed from the map, so this
// is a no-op for the happy path. Whatever output streamed before termination is
// still flushed, so partial generations remain visible in the trace.
func (h *OTelHook) OnStreamEnd(ctx context.Context, meta litellm.CallMeta, err error) {
	defer recoverHook()
	st := h.take(meta.CallID)
	if st == nil {
		return
	}
	if err != nil {
		st.span.RecordError(err)
		st.span.SetStatus(codes.Error, err.Error())
	}
	if h.captureContent && st.content.Len() > 0 {
		st.span.SetAttributes(attribute.String(attrCompletion, st.content.String()))
	}
	st.span.End()
}

// take removes and returns the call state for id, or nil if absent.
func (h *OTelHook) take(id string) *callState {
	h.mu.Lock()
	defer h.mu.Unlock()
	st := h.spans[id]
	delete(h.spans, id)
	return st
}

// stampResponse records the response-side attributes shared by the streaming
// and non-streaming paths. usage may be nil.
func stampResponse(span trace.Span, model, finishReason string, usage *litellm.Usage) {
	if model != "" {
		span.SetAttributes(attribute.String(attrResponseModel, model))
	}
	if finishReason != "" {
		span.SetAttributes(attribute.String(attrFinishReason, finishReason))
	}
	if usage != nil {
		span.SetAttributes(
			attribute.Int(attrInputTokens, usage.PromptTokens),
			attribute.Int(attrOutputTokens, usage.CompletionTokens),
		)
		if usage.CacheReadInputTokens > 0 {
			span.SetAttributes(attribute.Int(attrCacheReadTokens, usage.CacheReadInputTokens))
		}
	}
}

func recoverHook() {
	_ = recover() // observability must never break the LLM call
}
