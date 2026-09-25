// Package otel adapts litellm observers to OpenTelemetry generation spans.
package otel

import (
	"context"

	"github.com/voocel/litellm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// Observer is immutable after construction and may observe concurrent calls.
// Each Start owns its span; no global call registry or content collector is used.
type Observer struct {
	tracer         trace.Tracer
	captureContent bool
	attrFn         func(context.Context) []attribute.KeyValue
}

var _ litellm.Observer = (*Observer)(nil)

// Start implements litellm.Observer, starting one span per call.
func (o *Observer) Start(ctx context.Context, info litellm.CallInfo) (next context.Context, call litellm.CallObserver) {
	// Preserve the existing adapter's panic isolation. If setup panics after
	// creating a span, close it; the SDK still receives its original context.
	next = ctx
	var span trace.Span
	defer func() {
		if recover() != nil {
			if span != nil {
				span.End()
			}
			next = ctx
			call = nil
		}
	}()
	operation := semanticOperation(info.Provider)
	var model string
	if info.Request != nil {
		model = info.Request.Model
	}
	attrs := []attribute.KeyValue{
		attribute.String(attrProviderName, semanticProvider(info.Provider)),
		attribute.String(attrOperationName, operation),
		attribute.String(attrRequestModel, model),
	}
	if info.Streaming {
		attrs = append(attrs, attribute.Bool(attrRequestStream, true))
	}
	if o.captureContent && info.Request != nil && len(info.Request.Messages) > 0 {
		if data, err := marshalInputMessages(info.Request.Messages); err == nil {
			attrs = append(attrs, attribute.String(attrInputMessages, data))
		}
	}
	if o.attrFn != nil {
		attrs = append(attrs, o.attrFn(ctx)...)
	}
	name := operation
	if model != "" {
		name += " " + model
	}
	next, span = o.tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	return next, &observation{span: span, captureContent: o.captureContent}
}

type observation struct {
	span           trace.Span
	captureContent bool
}

func (o *observation) OnEvent(litellm.Event) {}
func (o *observation) End(result litellm.CallResult) {
	defer recoverObserver()
	defer o.span.End()
	o.span.SetAttributes(attribute.String("litellm.call.status", string(result.Status)))
	if result.Err != nil {
		recordSpanError(o.span, result.Err)
	}
	if resp := result.Response; resp != nil {
		finish := resp.FinishReason
		if result.Status != litellm.CallCompleted {
			finish = ""
		}
		stampResponse(o.span, resp.Model, string(finish), &resp.Usage)
		if o.captureContent && len(resp.Blocks) > 0 {
			setOutputMessages(o.span, resp.Blocks, finish)
		}
	}
}

// stampResponse records the response-side attributes shared by the streaming
// and non-streaming paths. usage may be nil.
func stampResponse(span trace.Span, model, finishReason string, usage *litellm.Usage) {
	if model != "" {
		span.SetAttributes(attribute.String(attrResponseModel, model))
	}
	if finishReason != "" {
		span.SetAttributes(attribute.StringSlice(attrFinishReasons, []string{semanticFinishReason(litellm.FinishReason(finishReason))}))
	}
	if usage != nil {
		for _, count := range []struct {
			key   string
			value *int
		}{
			{attrInputTokens, usage.InputTokens},
			{attrOutputTokens, usage.OutputTokens},
			{attrCacheReadTokens, usage.CacheReadTokens},
			{attrCacheWriteTokens, usage.CacheWriteTokens},
			{attrReasoningTokens, usage.ReasoningTokens},
		} {
			if count.value != nil {
				span.SetAttributes(attribute.Int(count.key, *count.value))
			}
		}
	}

}

func setOutputMessages(span trace.Span, blocks []litellm.Block, finishReason litellm.FinishReason) {
	if data, err := marshalOutputMessages(blocks, finishReason); err == nil {
		span.SetAttributes(attribute.String(attrOutputMessages, data))
	}
}

func recordSpanError(span trace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(attribute.String(attrErrorType, semanticErrorType(err)))
}

func recoverObserver() {
	_ = recover() // observability must never break the LLM call
}
