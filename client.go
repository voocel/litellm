package litellm

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/voocel/litellm/internal/idle"
)

// Client runs requests through a Provider. It copies and validates each
// request, attributes errors to the provider, and notifies Observers. A Client
// is safe for concurrent use.
type Client struct {
	provider           Provider
	observers          []Observer
	captureRawResponse bool
	streamIdleTimeout  time.Duration
}

// ClientOption configures a Client.
type ClientOption func(*Client) error

// New returns a Client for provider.
func New(provider Provider, opts ...ClientOption) (*Client, error) {
	if provider == nil {
		return nil, fmt.Errorf("provider cannot be nil")
	}
	client := &Client{provider: provider}
	for _, opt := range opts {
		if err := opt(client); err != nil {
			return nil, fmt.Errorf("apply client option: %w", err)
		}
	}
	return client, nil
}

// WithCaptureRawResponse keeps the raw vendor body in Response.Raw.
func WithCaptureRawResponse(enabled bool) ClientOption {
	return func(c *Client) error {
		c.captureRawResponse = enabled
		return nil
	}
}

// WithStreamIdleTimeout fails a stream, with a temporary network error, once
// it waits timeout for data from its connection, as when the connection hung:
// for the response, the waits of any retries its HTTP client makes included,
// or then for its body. Any data counts, vendor pings and gateway heartbeats
// included, so a healthy stream fails only when silent for longer, as a
// vendor may be while the model reads a long prompt or thinks. Zero disables
// the check.
func WithStreamIdleTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) error {
		if timeout < 0 {
			return fmt.Errorf("stream idle timeout cannot be negative")
		}
		c.streamIdleTimeout = timeout
		return nil
	}
}

// ProviderName returns the provider's Name.
func (c *Client) ProviderName() string {
	if c == nil || c.provider == nil {
		return ""
	}
	return c.provider.Name()
}

// Capabilities reports what the provider adapter can express on the wire; ok
// is false when the provider does not implement CapabilityProvider, so the
// facts are unknown.
func (c *Client) Capabilities() (caps Capabilities, ok bool) {
	if c == nil {
		return Capabilities{}, false
	}
	cp, ok := c.provider.(CapabilityProvider)
	if !ok {
		return Capabilities{}, false
	}
	caps = cp.Capabilities()
	caps.ProviderOptions = slices.Clone(caps.ProviderOptions)
	return caps, true
}

// Chat sends req and returns the complete response. On error the response
// may still be returned when the provider produced one.
func (c *Client) Chat(ctx context.Context, req Request) (*Response, error) {
	ctx, call := c.startCall(ctx, req, false)
	prepared, err := prepareRequest(req)
	if err != nil {
		call.end(callStatus(err), nil, err)
		return nil, err
	}
	resp, err := c.provider.Chat(ctx, prepared)
	if err != nil {
		err = endedBy(ctx, c.provider.Name(), WrapError(c.provider.Name(), ErrorTypeProvider, err))
	}
	if err == nil {
		err = validateResponse(resp, c.provider.Name(), prepared.Model)
	}
	if resp != nil {
		if !c.captureRawResponse {
			resp.Raw = nil
		}
		finalizeResponse(resp, c.provider.Name(), prepared.Model)
		for _, warning := range resp.Warnings {
			call.event(WarningEvent{Warning: warning})
		}
	}
	call.end(callStatus(err), resp, err)
	return resp, err
}

// Stream sends req and returns its event stream, which the caller must Close.
// Consume it with Next, Handle or Collect.
func (c *Client) Stream(ctx context.Context, req Request) (Stream, error) {
	streamCtx, cancel := context.WithCancel(ctx)
	if c.streamIdleTimeout > 0 {
		streamCtx = idle.WithTimeout(streamCtx, c.streamIdleTimeout)
	}
	streamCtx, call := c.startCall(streamCtx, req, true)
	prepared, err := prepareRequest(req)
	if err != nil {
		call.end(callStatus(err), nil, err)
		cancel()
		return nil, err
	}
	stream, err := c.provider.Stream(streamCtx, prepared)
	if err != nil {
		err = endedBy(streamCtx, c.provider.Name(), WrapError(c.provider.Name(), ErrorTypeProvider, err))
	} else if stream == nil {
		err = NewError(c.provider.Name(), ErrorTypeInternal, "provider returned nil stream without error", nil)
	}
	if err != nil {
		call.end(callStatus(err), nil, err)
		cancel()
		return nil, err
	}
	stream = newValidatedStream(c.provider.Name(), prepared.Model, stream)
	return &observedStream{ctx: streamCtx, cancel: cancel, call: call, provider: c.provider.Name(), inner: stream}, nil
}

// endedBy is err as the caller sees it once ctx has ended: the call was
// canceled or ran out of time, whatever err says. A request fails with the
// cause its context ended with, which need not be context.Canceled.
func endedBy(ctx context.Context, provider string, err error) error {
	ended := ctx.Err()
	if ended == nil || isContextError(err) {
		return err
	}
	message := err.Error()
	if e, ok := errors.AsType[*Error](err); ok {
		message = e.Message
	}
	cause := context.Cause(ctx)
	if cause != ended {
		cause = fmt.Errorf("%w: %w", ended, cause)
	}
	return NewNetworkError(provider, message, cause)
}

func prepareRequest(req Request) (*Request, error) {
	prepared := cloneRequest(req)
	if err := validateRequest(prepared); err != nil {
		return nil, err
	}
	return prepared, nil
}
