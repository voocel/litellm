package litellm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"
)

// Client is the main LLM client
type Client struct {
	providers map[string]Provider
	defaults  DefaultConfig
	router    Router
	mu        sync.RWMutex
}

// DefaultConfig holds default configuration values
type DefaultConfig struct {
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
	Resilience  ResilienceConfig `json:"resilience"`
}

// ClientOption defines options for configuring the client
type ClientOption func(*Client) error

// New creates a new LiteLLM client with optional configuration
func New(opts ...ClientOption) (*Client, error) {
	client := &Client{
		providers: make(map[string]Provider),
		defaults: DefaultConfig{
			MaxTokens:   4096,
			Temperature: 0.7,
			Resilience:  DefaultResilienceConfig(),
		},
		router: DefaultRouter, // Use default smart router
	}

	// If no options provided, use auto-discovery
	if len(opts) == 0 {
		if err := client.autoDiscoverProviders(); err != nil {
			return nil, err
		}
	} else {
		for _, opt := range opts {
			if err := opt(client); err != nil {
				return nil, fmt.Errorf("failed to apply option: %w", err)
			}
		}
	}

	// Validate that at least one provider is configured
	if len(client.providers) == 0 {
		return nil, fmt.Errorf("no providers configured: set environment variables (e.g., OPENAI_API_KEY) or use WithOpenAI()/WithAnthropic() options")
	}

	return client, nil
}

func (c *Client) addProviderFromConfig(name string, config ProviderConfig) error {
	provider, err := createProvider(name, config)
	if err != nil {
		return fmt.Errorf("%s provider: %w", name, err)
	}
	if err := provider.Validate(); err != nil {
		return fmt.Errorf("%s provider validation: %w", name, err)
	}
	c.providers[name] = provider
	return nil
}

// WithDefaults sets default configuration values
func WithDefaults(maxTokens int, temperature float64) ClientOption {
	return func(c *Client) error {
		c.defaults.MaxTokens = maxTokens
		c.defaults.Temperature = temperature
		return nil
	}
}

// WithResilience sets default resilience configuration
func WithResilience(config ResilienceConfig) ClientOption {
	return func(c *Client) error {
		c.defaults.Resilience = config
		return nil
	}
}

// WithTimeout sets request timeout for all providers
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) error {
		c.defaults.Resilience.RequestTimeout = timeout
		return nil
	}
}

// WithRetries sets retry configuration for all providers
func WithRetries(maxRetries int, initialDelay time.Duration) ClientOption {
	return func(c *Client) error {
		c.defaults.Resilience.MaxRetries = maxRetries
		c.defaults.Resilience.InitialDelay = initialDelay
		return nil
	}
}

// WithRouter sets a custom router for provider selection
func WithRouter(router Router) ClientOption {
	return func(c *Client) error {
		c.router = router
		return nil
	}
}

// WithOpenAI adds OpenAI provider with custom configuration
func WithOpenAI(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("openai", config)
	}
}

// WithAnthropic adds Anthropic provider with custom configuration
func WithAnthropic(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("anthropic", config)
	}
}

// WithGemini adds Gemini provider with custom configuration
func WithGemini(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("gemini", config)
	}
}

// WithDeepSeek adds DeepSeek provider with custom configuration
func WithDeepSeek(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("deepseek", config)
	}
}

// WithOpenRouter adds OpenRouter provider with custom configuration
func WithOpenRouter(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("openrouter", config)
	}
}

// WithQwen adds Qwen provider with custom configuration
func WithQwen(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("qwen", config)
	}
}

// WithGLM adds GLM provider with custom configuration
func WithGLM(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) error {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		return c.addProviderFromConfig("glm", config)
	}
}

// WithBedrock adds AWS Bedrock provider with custom configuration
// accessKeyID and secretAccessKey are AWS credentials
// regions is optional (defaults to "us-east-1")
func WithBedrock(accessKeyID, secretAccessKey string, region ...string) ClientOption {
	return func(c *Client) error {
		r := "us-east-1"
		if len(region) > 0 && region[0] != "" {
			r = region[0]
		}
		config := ProviderConfig{
			Resilience: c.defaults.Resilience,
			Extra: map[string]any{
				"access_key_id":     accessKeyID,
				"secret_access_key": secretAccessKey,
				"region":            r,
			},
		}
		return c.addProviderFromConfig("bedrock", config)
	}
}

// WithProvider adds a custom provider
func WithProvider(name string, provider Provider) ClientOption {
	return func(c *Client) error {
		if err := provider.Validate(); err != nil {
			return fmt.Errorf("%s provider validation: %w", name, err)
		}
		c.providers[name] = provider
		return nil
	}
}

// WithProviderConfig adds a provider using ProviderConfig
func WithProviderConfig(name string, config ProviderConfig) ClientOption {
	return func(c *Client) error {
		return c.addProviderFromConfig(name, config)
	}
}

// Chat performs a completion request
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

	provider, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	c.applyDefaults(req)

	resp, err := provider.Chat(ctx, req)
	if err != nil {
		return nil, WrapError(err, provider.Name())
	}
	return resp, nil
}

// Stream performs a streaming completion request
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

	provider, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	c.applyDefaults(req)

	stream, err := provider.Stream(ctx, req)
	if err != nil {
		return nil, WrapError(err, provider.Name())
	}
	return stream, nil
}

// Models returns all available models
func (c *Client) Models() []ModelInfo {
	c.mu.RLock()
	providers := make([]Provider, 0, len(c.providers))
	for _, p := range c.providers {
		providers = append(providers, p)
	}
	c.mu.RUnlock()

	var allModels []ModelInfo
	for _, provider := range providers {
		models := provider.Models()
		for _, model := range models {
			if model.Provider == "" {
				model.Provider = provider.Name()
			}
			allModels = append(allModels, model)
		}
	}
	return allModels
}

// Providers returns the names of all configured providers
func (c *Client) Providers() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var names []string
	for name := range c.providers {
		names = append(names, name)
	}
	return names
}

// AddProvider adds a provider to the client
func (c *Client) AddProvider(name string, provider Provider) error {
	if err := provider.Validate(); err != nil {
		return fmt.Errorf("provider validation failed: %w", err)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.providers[name] = provider
	return nil
}

// providerEnvConfig defines environment variable mapping for auto-discovery
type providerEnvConfig struct {
	name       string
	keyEnv     string
	urlEnv     string
	defaultURL string
}

// standardProviders lists all API-key based providers for auto-discovery
var standardProviders = []providerEnvConfig{
	{"openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com"},
	{"anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "https://api.anthropic.com"},
	{"gemini", "GEMINI_API_KEY", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"},
	{"deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"},
	{"openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"},
	{"qwen", "QWEN_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"},
	{"glm", "GLM_API_KEY", "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"},
}

// autoDiscoverProviders automatically discovers and configures providers from environment variables
func (c *Client) autoDiscoverProviders() error {
	var errs []error
	for _, p := range standardProviders {
		if apiKey := os.Getenv(p.keyEnv); apiKey != "" {
			config := ProviderConfig{
				APIKey:     apiKey,
				BaseURL:    getEnvOrDefault(p.urlEnv, p.defaultURL),
				Resilience: c.defaults.Resilience,
			}
			if err := c.addProviderFromConfig(p.name, config); err != nil {
				errs = append(errs, err)
			}
		}
	}

	// Discover AWS Bedrock (uses AWS credentials instead of API key)
	if accessKeyID := os.Getenv("AWS_ACCESS_KEY_ID"); accessKeyID != "" {
		if secretAccessKey := os.Getenv("AWS_SECRET_ACCESS_KEY"); secretAccessKey != "" {
			config := ProviderConfig{
				Resilience: c.defaults.Resilience,
				Extra: map[string]any{
					"access_key_id":     accessKeyID,
					"secret_access_key": secretAccessKey,
					"region":            getEnvOrDefault("AWS_REGION", "us-east-1"),
					"session_token":     os.Getenv("AWS_SESSION_TOKEN"),
				},
			}
			if err := c.addProviderFromConfig("bedrock", config); err != nil {
				errs = append(errs, err)
			}
		}
	}

	if len(errs) > 0 && len(c.providers) == 0 {
		return fmt.Errorf("auto-discovery failed: no providers could be configured. Errors: %w", errors.Join(errs...))
	}

	return nil
}

// resolveProvider resolves the provider for a model using the configured router
func (c *Client) resolveProvider(model string) (Provider, error) {
	c.mu.RLock()
	if len(c.providers) == 0 {
		c.mu.RUnlock()
		return nil, NewError(ErrorTypeValidation, "no providers configured: set environment variables (e.g., OPENAI_API_KEY) or use WithOpenAI()/WithAnthropic() options")
	}

	availableProviders := make([]Provider, 0, len(c.providers))
	for _, provider := range c.providers {
		availableProviders = append(availableProviders, provider)
	}
	c.mu.RUnlock()

	// Make routing deterministic in multi-provider scenarios (avoid random map iteration order).
	sort.Slice(availableProviders, func(i, j int) bool {
		return availableProviders[i].Name() < availableProviders[j].Name()
	})

	// Use configured router for provider selection
	return c.router.Route(model, availableProviders)
}

// applyDefaults applies default configuration to the request
func (c *Client) applyDefaults(req *Request) {
	if req.MaxTokens == nil {
		maxTokens := c.defaults.MaxTokens
		req.MaxTokens = &maxTokens
	}
	if req.Temperature == nil {
		temperature := c.defaults.Temperature
		req.Temperature = &temperature
	}
}

// getEnvOrDefault returns environment variable value or default
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

var (
	// defaultClient is the singleton client used by Quick functions
	defaultClient     *Client
	defaultClientOnce sync.Once
	defaultClientErr  error
)

// getDefaultClient returns the singleton default client, creating it if necessary
func getDefaultClient() (*Client, error) {
	defaultClientOnce.Do(func() {
		defaultClient, defaultClientErr = New()
	})
	return defaultClient, defaultClientErr
}

// ResetDefaultClient resets the default client singleton
// This is useful for testing or when environment variables change
func ResetDefaultClient() {
	defaultClientOnce = sync.Once{}
	defaultClient = nil
	defaultClientErr = nil
}

// Quick performs a quick completion with minimal configuration
// It uses a singleton client with auto-discovery and makes a simple completion request
// with a default timeout of 30 seconds.
//
// The client is created once on first call and reused for subsequent calls,
// providing better performance through connection pooling.
func Quick(model, message string) (*Response, error) {
	return QuickWithTimeout(model, message, 30*time.Second)
}

// QuickWithTimeout performs a quick completion with a custom timeout
// It uses a singleton client with auto-discovery and makes a simple completion request.
//
// The client is created once on first call and reused for subsequent calls,
// providing better performance through connection pooling.
func QuickWithTimeout(model, message string, timeout time.Duration) (*Response, error) {
	if model == "" {
		return nil, fmt.Errorf("litellm.Quick: model parameter cannot be empty")
	}
	if message == "" {
		return nil, fmt.Errorf("litellm.Quick: message parameter cannot be empty")
	}
	if timeout <= 0 {
		return nil, fmt.Errorf("litellm.Quick: timeout must be positive, got %v", timeout)
	}

	// Get or create singleton client with auto-discovery
	client, err := getDefaultClient()
	if err != nil {
		return nil, fmt.Errorf("litellm.Quick: failed to initialize client (check API keys in environment): %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	resp, err := client.Chat(ctx, &Request{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: message},
		},
	})
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, fmt.Errorf("litellm.Quick: request timed out after %v (model: %s)", timeout, model)
		}
		return nil, fmt.Errorf("litellm.Quick: request failed for model %s: %w", model, err)
	}

	return resp, nil
}
