package litellm

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

// Client is a minimal, predictable client bound to a single Provider.
type Client struct {
	provider Provider
	defaults DefaultConfig
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

	c.debugRequest(&reqCopy, false)
	start := time.Now()

	resp, err := c.provider.Chat(ctx, &reqCopy)

	c.debugResponse(resp, err, time.Since(start))

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

	c.debugRequest(&reqCopy, true)
	start := time.Now()

	stream, err := c.provider.Stream(ctx, &reqCopy)
	if err != nil {
		c.debugStreamError(err, time.Since(start))
		return nil, WrapError(err, c.provider.Name())
	}

	c.debugStreamReady(time.Since(start))
	return stream, nil
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

	c.debugResponsesRequest(&reqCopy)
	start := time.Now()

	provider, ok := c.provider.(interface {
		Responses(context.Context, *OpenAIResponsesRequest) (*Response, error)
	})
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}

	resp, err := provider.Responses(ctx, &reqCopy)

	c.debugResponse(resp, err, time.Since(start))

	return resp, err
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

	c.debugResponsesRequest(&reqCopy)
	start := time.Now()

	provider, ok := c.provider.(interface {
		ResponsesStream(context.Context, *OpenAIResponsesRequest) (StreamReader, error)
	})
	if !ok {
		return nil, NewError(ErrorTypeValidation, "responses API is only supported by the OpenAI provider")
	}

	stream, err := provider.ResponsesStream(ctx, &reqCopy)
	if err != nil {
		c.debugStreamError(err, time.Since(start))
		return nil, err
	}

	c.debugStreamReady(time.Since(start))
	return stream, nil
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

// debugLog writes a debug message if debug mode is enabled.
func (c *Client) debugLog(format string, args ...any) {
	if !c.debug || c.debugOut == nil {
		return
	}
	fmt.Fprintf(c.debugOut, "[litellm:%s] "+format+"\n", append([]any{c.provider.Name()}, args...)...)
}

// debugRequest logs request details.
func (c *Client) debugRequest(req *Request, stream bool) {
	if !c.debug {
		return
	}
	mode := "chat"
	if stream {
		mode = "stream"
	}
	c.debugLog("→ %s model=%s messages=%d", mode, req.Model, len(req.Messages))
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

// debugResponse logs response details.
func (c *Client) debugResponse(resp *Response, err error, duration time.Duration) {
	if !c.debug {
		return
	}
	if err != nil {
		c.debugLog("← error (%v): %v", duration.Round(time.Millisecond), err)
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

// debugStreamError logs stream error.
func (c *Client) debugStreamError(err error, duration time.Duration) {
	if !c.debug {
		return
	}
	c.debugLog("← stream error (%v): %v", duration.Round(time.Millisecond), err)
}

// debugStreamReady logs stream ready.
func (c *Client) debugStreamReady(duration time.Duration) {
	if !c.debug {
		return
	}
	c.debugLog("← stream ready (%v)", duration.Round(time.Millisecond))
}

// debugResponsesRequest logs OpenAI Responses API request details.
func (c *Client) debugResponsesRequest(req *OpenAIResponsesRequest) {
	if !c.debug {
		return
	}
	c.debugLog("→ responses model=%s messages=%d", req.Model, len(req.Messages))
	if req.MaxOutputTokens != nil {
		c.debugLog("  max_output_tokens=%d", *req.MaxOutputTokens)
	}
	if req.ReasoningEffort != "" {
		c.debugLog("  reasoning_effort=%s", req.ReasoningEffort)
	}
}
