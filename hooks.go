package litellm

import (
	"context"
	"time"
)

// CallMeta describes a single client call.
type CallMeta struct {
	CallID    string
	Provider  string
	Operation string
	Model     string
	Streaming bool
	StartedAt time.Time
	Duration  time.Duration
}

// Hook observes request execution without modifying control flow.
// For streaming calls, AfterResponse runs when the stream is established; resp is nil in that case.
type Hook interface {
	BeforeRequest(ctx context.Context, meta CallMeta)
	AfterResponse(ctx context.Context, meta CallMeta, resp *Response, err error)
	OnStreamChunk(ctx context.Context, meta CallMeta, chunk *StreamChunk)
}

// HookFuncs adapts plain functions into a Hook.
type HookFuncs struct {
	BeforeRequestFunc func(ctx context.Context, meta CallMeta)
	AfterResponseFunc func(ctx context.Context, meta CallMeta, resp *Response, err error)
	OnStreamChunkFunc func(ctx context.Context, meta CallMeta, chunk *StreamChunk)
}

func (h HookFuncs) BeforeRequest(ctx context.Context, meta CallMeta) {
	if h.BeforeRequestFunc != nil {
		h.BeforeRequestFunc(ctx, meta)
	}
}

func (h HookFuncs) AfterResponse(ctx context.Context, meta CallMeta, resp *Response, err error) {
	if h.AfterResponseFunc != nil {
		h.AfterResponseFunc(ctx, meta, resp, err)
	}
}

func (h HookFuncs) OnStreamChunk(ctx context.Context, meta CallMeta, chunk *StreamChunk) {
	if h.OnStreamChunkFunc != nil {
		h.OnStreamChunkFunc(ctx, meta, chunk)
	}
}

// WithHook appends a single execution hook to the client.
func WithHook(h Hook) ClientOption {
	return func(c *Client) error {
		if h == nil {
			return nil
		}
		c.hooks = append(c.hooks, h)
		return nil
	}
}

// WithHooks appends multiple execution hooks to the client.
func WithHooks(hooks ...Hook) ClientOption {
	return func(c *Client) error {
		for _, h := range hooks {
			if h == nil {
				continue
			}
			c.hooks = append(c.hooks, h)
		}
		return nil
	}
}
