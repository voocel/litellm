package litellm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/voocel/litellm/providers"
)

type requestCallOptions struct {
	operation string
}

type responsesCallOptions struct {
	operation string
}

type responsesProvider interface {
	Responses(context.Context, *OpenAIResponsesRequest) (*Response, error)
}

type responsesStreamProvider interface {
	ResponsesStream(context.Context, *OpenAIResponsesRequest) (StreamReader, error)
}

var callIDSeq atomic.Uint64

func (c *Client) prepareRequest(req *Request) (*Request, error) {
	if req == nil {
		return nil, NewError(ErrorTypeValidation, "request cannot be nil")
	}
	if req.Model == "" {
		return nil, NewError(ErrorTypeValidation, "model cannot be empty")
	}
	if len(req.Messages) == 0 {
		return nil, NewError(ErrorTypeValidation, "messages cannot be empty")
	}

	reqCopy := *req
	c.applyDefaults(&reqCopy)
	if err := validateRequestMessages(c.provider.Name(), reqCopy.Messages); err != nil {
		return nil, err
	}

	return &reqCopy, nil
}

func (c *Client) prepareResponsesRequest(req *OpenAIResponsesRequest) (*OpenAIResponsesRequest, error) {
	if req == nil {
		return nil, NewError(ErrorTypeValidation, "responses request cannot be nil")
	}
	if req.Model == "" {
		return nil, NewError(ErrorTypeValidation, "model cannot be empty")
	}
	if len(req.Messages) == 0 {
		return nil, NewError(ErrorTypeValidation, "messages cannot be empty")
	}

	reqCopy := *req
	c.applyResponsesDefaults(&reqCopy)
	if err := validateRequestMessages(c.provider.Name(), reqCopy.Messages); err != nil {
		return nil, err
	}

	return &reqCopy, nil
}

func (c *Client) applyDefaults(req *Request) {
	if req.MaxTokens == nil {
		maxTokens := c.defaults.MaxTokens
		req.MaxTokens = &maxTokens
	}
	if req.Temperature == nil {
		temperature := c.defaults.Temperature
		req.Temperature = &temperature
	}
	if req.TopP == nil {
		topP := c.defaults.TopP
		req.TopP = &topP
	}
}

func (c *Client) applyResponsesDefaults(req *OpenAIResponsesRequest) {
	if req.MaxOutputTokens == nil {
		maxTokens := c.defaults.MaxTokens
		req.MaxOutputTokens = &maxTokens
	}
	if req.Temperature == nil {
		temperature := c.defaults.Temperature
		req.Temperature = &temperature
	}
	if req.TopP == nil {
		topP := c.defaults.TopP
		req.TopP = &topP
	}
}

func validateToolCalls(provider string, toolCalls []ToolCall) error {
	if len(toolCalls) == 0 {
		return nil
	}

	seen := make(map[string]string, len(toolCalls))
	for _, toolCall := range toolCalls {
		if toolCall.ID == "" {
			continue
		}
		if prev, ok := seen[toolCall.ID]; ok {
			message := fmt.Sprintf("invalid tool calls: duplicate tool_call.id %q", toolCall.ID)
			if prev != "" && toolCall.Function.Name != "" && prev != toolCall.Function.Name {
				message = fmt.Sprintf("%s reused by tools %q and %q", message, prev, toolCall.Function.Name)
			}
			if provider != "" {
				return NewValidationError(provider, message)
			}
			return NewError(ErrorTypeValidation, message)
		}
		seen[toolCall.ID] = toolCall.Function.Name
	}

	return nil
}

func validateRequestMessages(provider string, messages []Message) error {
	for i, msg := range messages {
		if msg.Role != "assistant" || msg.IsError || len(msg.ToolCalls) == 0 {
			continue
		}

		seen := make(map[string]string, len(msg.ToolCalls))
		for _, toolCall := range msg.ToolCalls {
			normalizedID := providers.NormalizeToolCallID(toolCall.ID)
			if normalizedID == "" {
				continue
			}
			if prev, ok := seen[normalizedID]; ok {
				message := fmt.Sprintf("invalid request messages: assistant message %d has duplicate tool_call.id %q", i, normalizedID)
				if prev != "" && toolCall.Function.Name != "" && prev != toolCall.Function.Name {
					message = fmt.Sprintf("%s reused by tools %q and %q", message, prev, toolCall.Function.Name)
				}
				if provider != "" {
					return NewValidationError(provider, message)
				}
				return NewError(ErrorTypeValidation, message)
			}
			seen[normalizedID] = toolCall.Function.Name
		}
	}
	return nil
}

func (c *Client) executeRequestCall(
	ctx context.Context,
	req *Request,
	opts requestCallOptions,
	invoke func(context.Context, *Request) (*Response, error),
) (*Response, error) {
	prepared, err := c.prepareRequest(req)
	if err != nil {
		return nil, err
	}

	meta := c.newCallMeta(opts.operation, prepared.Model, false)
	c.notifyBeforeRequest(ctx, meta)
	c.debugRequest(prepared, opts.operation)
	start := meta.StartedAt

	resp, err := invoke(ctx, prepared)
	return c.finalizeResponseCall(ctx, meta, resp, err, time.Since(start))
}

func (c *Client) executeRequestStreamCall(
	ctx context.Context,
	req *Request,
	opts requestCallOptions,
	invoke func(context.Context, *Request) (StreamReader, error),
) (StreamReader, error) {
	prepared, err := c.prepareRequest(req)
	if err != nil {
		return nil, err
	}

	meta := c.newCallMeta(opts.operation, prepared.Model, true)
	c.notifyBeforeRequest(ctx, meta)
	c.debugRequest(prepared, opts.operation)
	start := meta.StartedAt

	streamCtx, watchdogCancel, idleTimeout := c.resolveStreamWatchdog(ctx)
	stream, err := invoke(streamCtx, prepared)
	stream = c.attachStreamWatchdog(stream, err, watchdogCancel, idleTimeout)
	return c.finalizeStreamCall(ctx, meta, stream, err, time.Since(start))
}

func (c *Client) executeResponsesCall(
	ctx context.Context,
	req *OpenAIResponsesRequest,
	opts responsesCallOptions,
	invoke func(context.Context, responsesProvider, *OpenAIResponsesRequest) (*Response, error),
) (*Response, error) {
	provider, err := c.getResponsesProvider()
	if err != nil {
		return nil, err
	}

	prepared, err := c.prepareResponsesRequest(req)
	if err != nil {
		return nil, err
	}

	meta := c.newCallMeta(opts.operation, prepared.Model, false)
	c.notifyBeforeRequest(ctx, meta)
	c.debugResponsesRequest(prepared, opts.operation)
	start := meta.StartedAt

	resp, err := invoke(ctx, provider, prepared)
	return c.finalizeResponseCall(ctx, meta, resp, err, time.Since(start))
}

func (c *Client) executeResponsesStreamCall(
	ctx context.Context,
	req *OpenAIResponsesRequest,
	opts responsesCallOptions,
	invoke func(context.Context, responsesStreamProvider, *OpenAIResponsesRequest) (StreamReader, error),
) (StreamReader, error) {
	provider, err := c.getResponsesStreamProvider()
	if err != nil {
		return nil, err
	}

	prepared, err := c.prepareResponsesRequest(req)
	if err != nil {
		return nil, err
	}

	meta := c.newCallMeta(opts.operation, prepared.Model, true)
	c.notifyBeforeRequest(ctx, meta)
	c.debugResponsesRequest(prepared, opts.operation)
	start := meta.StartedAt

	streamCtx, watchdogCancel, idleTimeout := c.resolveStreamWatchdog(ctx)
	stream, err := invoke(streamCtx, provider, prepared)
	stream = c.attachStreamWatchdog(stream, err, watchdogCancel, idleTimeout)
	return c.finalizeStreamCall(ctx, meta, stream, err, time.Since(start))
}

func (c *Client) getResponsesProvider() (responsesProvider, error) {
	provider, ok := c.provider.(responsesProvider)
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}
	return provider, nil
}

func (c *Client) getResponsesStreamProvider() (responsesStreamProvider, error) {
	provider, ok := c.provider.(responsesStreamProvider)
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}
	return provider, nil
}

func (c *Client) finalizeResponseCall(ctx context.Context, meta CallMeta, resp *Response, err error, duration time.Duration) (*Response, error) {
	meta.Duration = duration
	c.debugResponse(resp, err, duration)

	if err != nil {
		wrappedErr := c.attachErrorModel(WrapError(err, c.provider.Name()), meta.Model)
		c.notifyAfterResponse(ctx, meta, nil, wrappedErr)
		return nil, wrappedErr
	}
	if resp == nil {
		internalErr := c.attachErrorModel(NewError(ErrorTypeInternal, "provider returned nil response without error"), meta.Model)
		c.notifyAfterResponse(ctx, meta, nil, internalErr)
		return nil, internalErr
	}
	if err := validateToolCalls(c.provider.Name(), resp.ToolCalls); err != nil {
		err = c.attachErrorModel(err, meta.Model)
		c.notifyAfterResponse(ctx, meta, nil, err)
		return nil, err
	}
	c.notifyAfterResponse(ctx, meta, resp, nil)
	return resp, nil
}

func (c *Client) finalizeStreamCall(ctx context.Context, meta CallMeta, stream StreamReader, err error, duration time.Duration) (StreamReader, error) {
	meta.Duration = duration
	if err != nil {
		c.debugStreamError(err, duration)
		wrappedErr := c.attachErrorModel(WrapError(err, c.provider.Name()), meta.Model)
		c.notifyAfterResponse(ctx, meta, nil, wrappedErr)
		return nil, wrappedErr
	}
	if stream == nil {
		internalErr := c.attachErrorModel(NewError(ErrorTypeInternal, "provider returned nil stream without error"), meta.Model)
		c.debugStreamError(internalErr, duration)
		c.notifyAfterResponse(ctx, meta, nil, internalErr)
		return nil, internalErr
	}

	c.debugStreamReady(duration)
	c.notifyAfterResponse(ctx, meta, nil, nil)
	return newHookedStreamReader(ctx, meta, c.hooks, stream), nil
}

func (c *Client) newCallMeta(operation, model string, streaming bool) CallMeta {
	return CallMeta{
		CallID:    newCallID(),
		Provider:  c.provider.Name(),
		Operation: operation,
		Model:     model,
		Streaming: streaming,
		StartedAt: time.Now(),
	}
}

func newCallID() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), callIDSeq.Add(1))
}

func (c *Client) attachErrorModel(err error, model string) error {
	if err == nil || model == "" {
		return err
	}

	var llmErr *LiteLLMError
	if errors.As(err, &llmErr) && llmErr.Model == "" {
		llmErr.Model = model
	}
	return err
}

func (c *Client) notifyBeforeRequest(ctx context.Context, meta CallMeta) {
	for _, h := range c.hooks {
		h.BeforeRequest(ctx, meta)
	}
}

func (c *Client) notifyAfterResponse(ctx context.Context, meta CallMeta, resp *Response, err error) {
	for _, h := range c.hooks {
		h.AfterResponse(ctx, meta, resp, err)
	}
}

func (c *Client) debugLog(format string, args ...any) {
	if !c.debug || c.debugOut == nil {
		return
	}
	fmt.Fprintf(c.debugOut, "[litellm:%s] "+format+"\n", append([]any{c.provider.Name()}, args...)...)
}

func (c *Client) debugRequest(req *Request, operation string) {
	if !c.debug {
		return
	}
	if operation == "" {
		operation = "chat"
	}
	c.debugLog("→ %s model=%s messages=%d", operation, req.Model, len(req.Messages))
	if req.MaxTokens != nil {
		c.debugLog("  max_tokens=%d", *req.MaxTokens)
	}
	if req.Temperature != nil {
		c.debugLog("  temperature=%.2f", *req.Temperature)
	}
	if len(req.Tools) > 0 {
		toolNames := make([]string, len(req.Tools))
		for i, t := range req.Tools {
			toolNames[i] = t.Function.Name
		}
		c.debugLog("  tools=[%s]", strings.Join(toolNames, ", "))
	}
}

func (c *Client) debugResponse(resp *Response, err error, duration time.Duration) {
	if !c.debug {
		return
	}
	if err != nil {
		c.debugLog("← error (%v): %v", duration.Round(time.Millisecond), err)
		return
	}
	if resp == nil {
		c.debugLog("← error (%v): nil response without error", duration.Round(time.Millisecond))
		return
	}
	c.debugLog("← ok (%v) tokens=%d (prompt=%d, completion=%d)",
		duration.Round(time.Millisecond),
		resp.Usage.TotalTokens,
		resp.Usage.PromptTokens,
		resp.Usage.CompletionTokens,
	)
	if resp.FinishReason != "" {
		c.debugLog("  finish_reason=%s", resp.FinishReason)
	}
	if len(resp.ToolCalls) > 0 {
		c.debugLog("  tool_calls=%d", len(resp.ToolCalls))
	}
}

func (c *Client) debugStreamError(err error, duration time.Duration) {
	if !c.debug {
		return
	}
	c.debugLog("← stream error (%v): %v", duration.Round(time.Millisecond), err)
}

func (c *Client) debugStreamReady(duration time.Duration) {
	if !c.debug {
		return
	}
	c.debugLog("← stream ready (%v)", duration.Round(time.Millisecond))
}

func (c *Client) debugResponsesRequest(req *OpenAIResponsesRequest, operation string) {
	if !c.debug {
		return
	}
	if operation == "" {
		operation = "responses"
	}
	c.debugLog("→ %s model=%s messages=%d", operation, req.Model, len(req.Messages))
	if req.MaxOutputTokens != nil {
		c.debugLog("  max_output_tokens=%d", *req.MaxOutputTokens)
	}
	if req.ReasoningEffort != "" {
		c.debugLog("  reasoning_effort=%s", req.ReasoningEffort)
	}
}
