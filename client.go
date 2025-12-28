package litellm

import (
	"context"
	"fmt"
	"strings"
)

// Client is a minimal, predictable client bound to a single Provider.
type Client struct {
	provider Provider
	defaults DefaultConfig
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
	if err := validateProviderConfig(name, config); err != nil {
		return nil, err
	}
	provider, err := createProvider(name, config)
	if err != nil {
		return nil, err
	}
	return New(provider, opts...)
}

func validateProviderConfig(name string, config ProviderConfig) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("provider name cannot be empty")
	}

	switch strings.ToLower(strings.TrimSpace(name)) {
	case "bedrock":
		if config.APIKey != "" {
			parts := strings.SplitN(config.APIKey, ":", 2)
			if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
				return nil
			}
		}
		if config.Extra == nil {
			return fmt.Errorf("bedrock requires access_key_id and secret_access_key")
		}
		accessKeyID, _ := config.Extra["access_key_id"].(string)
		secretAccessKey, _ := config.Extra["secret_access_key"].(string)
		if accessKeyID == "" || secretAccessKey == "" {
			return fmt.Errorf("bedrock requires access_key_id and secret_access_key")
		}
		return nil
	default:
		if config.APIKey == "" {
			return fmt.Errorf("%s requires api key", strings.ToLower(strings.TrimSpace(name)))
		}
		return nil
	}
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

// Chat executes a non-streaming request.
func (c *Client) Chat(ctx context.Context, req *Request) (*Response, error) {
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

	resp, err := c.provider.Chat(ctx, &reqCopy)
	if err != nil {
		return nil, WrapError(err, c.provider.Name())
	}
	return resp, nil
}

// Stream executes a streaming request.
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error) {
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

	stream, err := c.provider.Stream(ctx, &reqCopy)
	if err != nil {
		return nil, WrapError(err, c.provider.Name())
	}
	return stream, nil
}

// Responses executes an OpenAI Responses API request on an OpenAI provider.
func (c *Client) Responses(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
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

	provider, ok := c.provider.(interface {
		Responses(context.Context, *OpenAIResponsesRequest) (*Response, error)
	})
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}
	return provider.Responses(ctx, &reqCopy)
}

// ResponsesStream executes a streaming OpenAI Responses API request.
func (c *Client) ResponsesStream(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
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

	provider, ok := c.provider.(interface {
		ResponsesStream(context.Context, *OpenAIResponsesRequest) (StreamReader, error)
	})
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}
	return provider.ResponsesStream(ctx, &reqCopy)
}

// applyDefaults applies defaults when request fields are unset.
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
