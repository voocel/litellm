package litellm

import (
	"context"
	"fmt"
	"io"
	"os"
)

// Client is a minimal, predictable client bound to a single Provider.
type Client struct {
	provider Provider
	defaults DefaultConfig
	hooks    []Hook
	debug    bool
	debugOut io.Writer
}

// DefaultConfig holds request-level defaults.
type DefaultConfig struct {
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

// ClientOption configures the Client.
type ClientOption func(*Client) error

// New creates a client with an explicit Provider (no implicit discovery).
func New(provider Provider, opts ...ClientOption) (*Client, error) {
	if provider == nil {
		return nil, fmt.Errorf("provider cannot be nil")
	}

	client := &Client{
		provider: provider,
		defaults: DefaultConfig{MaxTokens: 4096, Temperature: 1, TopP: 1.0},
	}

	for _, opt := range opts {
		if err := opt(client); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	if err := provider.Validate(); err != nil {
		return nil, fmt.Errorf("%s provider validation failed: %w", provider.Name(), err)
	}

	return client, nil
}

// NewWithProvider creates a client from provider name and config.
func NewWithProvider(name string, config ProviderConfig, opts ...ClientOption) (*Client, error) {
	if err := validateProviderName(name); err != nil {
		return nil, err
	}
	provider, err := createProvider(name, config)
	if err != nil {
		return nil, err
	}
	return New(provider, opts...)
}

func validateProviderName(name string) error {
	if normalizeProviderName(name) == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	return nil
}

// WithDefaults sets request-level defaults (applies only when fields are unset).
func WithDefaults(maxTokens int, temperature float64, topP float64) ClientOption {
	return func(c *Client) error {
		if maxTokens <= 0 {
			return fmt.Errorf("maxTokens must be positive")
		}
		if temperature < 0 || temperature > 2 {
			return fmt.Errorf("temperature must be between 0 and 2")
		}
		if topP < 0 || topP > 1 {
			return fmt.Errorf("topP must be between 0 and 1")
		}
		c.defaults.MaxTokens = maxTokens
		c.defaults.Temperature = temperature
		c.defaults.TopP = topP
		return nil
	}
}

// WithDebug enables debug logging to stderr.
func WithDebug(enabled bool) ClientOption {
	return func(c *Client) error {
		c.debug = enabled
		if enabled && c.debugOut == nil {
			c.debugOut = os.Stderr
		}
		return nil
	}
}

// WithDebugOutput enables debug logging to a custom writer.
// If w is nil, debug logging is disabled.
func WithDebugOutput(w io.Writer) ClientOption {
	return func(c *Client) error {
		if w == nil {
			c.debug = false
			c.debugOut = nil
			return nil
		}
		c.debug = true
		c.debugOut = w
		return nil
	}
}

// ProviderName returns the name of the bound provider.
func (c *Client) ProviderName() string {
	if c.provider == nil {
		return ""
	}
	return c.provider.Name()
}

// Chat executes a non-streaming request.
func (c *Client) Chat(ctx context.Context, req *Request) (*Response, error) {
	return c.executeRequestCall(ctx, req, requestCallOptions{
		operation: "chat",
	}, func(ctx context.Context, prepared *Request) (*Response, error) {
		return c.provider.Chat(ctx, prepared)
	})
}

// Stream executes a streaming request.
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	return c.executeRequestStreamCall(ctx, req, requestCallOptions{
		operation: "stream",
	}, func(ctx context.Context, prepared *Request) (StreamReader, error) {
		return c.provider.Stream(ctx, prepared)
	})
}

// ListModels returns the list of available models for the bound provider (if supported).
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if c == nil || c.provider == nil {
		return nil, NewError(ErrorTypeValidation, "client provider cannot be nil")
	}

	lister, ok := c.provider.(interface {
		ListModels(context.Context) ([]ModelInfo, error)
	})
	if !ok {
		return nil, NewError(ErrorTypeValidation, fmt.Sprintf("%s provider does not support model listing", c.provider.Name()))
	}

	models, err := lister.ListModels(ctx)
	if err != nil {
		return nil, WrapError(err, c.provider.Name())
	}
	return models, nil
}

// Responses executes an OpenAI Responses API request on an OpenAI provider.
func (c *Client) Responses(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
	return c.executeResponsesCall(ctx, req, responsesCallOptions{
		operation: "responses",
	}, func(ctx context.Context, provider responsesProvider, prepared *OpenAIResponsesRequest) (*Response, error) {
		return provider.Responses(ctx, prepared)
	})
}

// ResponsesStream executes a streaming OpenAI Responses API request.
func (c *Client) ResponsesStream(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
	return c.executeResponsesStreamCall(ctx, req, responsesCallOptions{
		operation: "responses_stream",
	}, func(ctx context.Context, provider responsesStreamProvider, prepared *OpenAIResponsesRequest) (StreamReader, error) {
		return provider.ResponsesStream(ctx, prepared)
	})
}
