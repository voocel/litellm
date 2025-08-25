package litellm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"
)

// Client is the main LLM client
type Client struct {
	providers map[string]Provider
	defaults  DefaultConfig
}

// DefaultConfig holds default configuration values
type DefaultConfig struct {
	MaxTokens   int              `json:"max_tokens"`
	Temperature float64          `json:"temperature"`
	Resilience  ResilienceConfig `json:"resilience"`
}

// ClientOption defines options for configuring the client
type ClientOption func(*Client)

// New creates a new LiteLLM client with optional configuration
func New(opts ...ClientOption) *Client {
	client := &Client{
		providers: make(map[string]Provider),
		defaults: DefaultConfig{
			MaxTokens:   4096,
			Temperature: 0.7,
			Resilience:  DefaultResilienceConfig(),
		},
	}

	// If no options provided, use auto-discovery
	if len(opts) == 0 {
		client.autoDiscoverProviders()
	} else {
		for _, opt := range opts {
			opt(client)
		}
	}

	return client
}

// WithDefaults sets default configuration values
func WithDefaults(maxTokens int, temperature float64) ClientOption {
	return func(c *Client) {
		c.defaults.MaxTokens = maxTokens
		c.defaults.Temperature = temperature
	}
}

// WithResilience sets default resilience configuration
func WithResilience(config ResilienceConfig) ClientOption {
	return func(c *Client) {
		c.defaults.Resilience = config
	}
}

// WithTimeout sets request timeout for all providers
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) {
		c.defaults.Resilience.RequestTimeout = timeout
	}
}

// WithRetries sets retry configuration for all providers
func WithRetries(maxRetries int, initialDelay time.Duration) ClientOption {
	return func(c *Client) {
		c.defaults.Resilience.MaxRetries = maxRetries
		c.defaults.Resilience.InitialDelay = initialDelay
	}
}

// WithOpenAI adds OpenAI provider with custom configuration
func WithOpenAI(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("openai", config); err == nil {
			c.providers["openai"] = provider
		}
	}
}

// WithAnthropic adds Anthropic provider with custom configuration
func WithAnthropic(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("anthropic", config); err == nil {
			c.providers["anthropic"] = provider
		}
	}
}

// WithGemini adds Gemini provider with custom configuration
func WithGemini(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("gemini", config); err == nil {
			c.providers["gemini"] = provider
		}
	}
}

// WithDeepSeek adds DeepSeek provider with custom configuration
func WithDeepSeek(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("deepseek", config); err == nil {
			c.providers["deepseek"] = provider
		}
	}
}

// WithOpenRouter adds OpenRouter provider with custom configuration
func WithOpenRouter(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("openrouter", config); err == nil {
			c.providers["openrouter"] = provider
		}
	}
}

// WithQwen adds Qwen provider with custom configuration
func WithQwen(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("qwen", config); err == nil {
			c.providers["qwen"] = provider
		}
	}
}

// WithGLM adds GLM provider with custom configuration
func WithGLM(apiKey string, baseURL ...string) ClientOption {
	return func(c *Client) {
		config := ProviderConfig{
			APIKey:     apiKey,
			Resilience: c.defaults.Resilience,
		}
		if len(baseURL) > 0 && baseURL[0] != "" {
			config.BaseURL = baseURL[0]
		}
		if provider, err := createProvider("glm", config); err == nil {
			c.providers["glm"] = provider
		}
	}
}

// WithProvider adds a custom provider
func WithProvider(name string, provider Provider) ClientOption {
	return func(c *Client) {
		if err := provider.Validate(); err == nil {
			c.providers[name] = provider
		}
	}
}

// WithProviderConfig adds a provider using ProviderConfig
func WithProviderConfig(name string, config ProviderConfig) ClientOption {
	return func(c *Client) {
		if provider, err := createProvider(name, config); err == nil {
			if err := provider.Validate(); err == nil {
				c.providers[name] = provider
			}
		}
	}
}

// Chat performs a completion request
func (c *Client) Chat(ctx context.Context, req *Request) (*Response, error) {
	provider, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	c.applyDefaults(req)

	return provider.Chat(ctx, req)
}

// Stream performs a streaming completion request
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	provider, err := c.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	c.applyDefaults(req)

	return provider.Stream(ctx, req)
}

// Models returns all available models
func (c *Client) Models() []ModelInfo {
	var allModels []ModelInfo
	for _, provider := range c.providers {
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
	c.providers[name] = provider
	return nil
}

// autoDiscoverProviders automatically discovers and configures providers from environment variables
func (c *Client) autoDiscoverProviders() {
	// Auto-discover OpenAI
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("OPENAI_BASE_URL", "https://api.openai.com"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("openai", config); err == nil {
			c.providers["openai"] = provider
		}
	}

	// Auto-discover Anthropic
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("anthropic", config); err == nil {
			c.providers["anthropic"] = provider
		}
	}

	// Auto-discover Gemini
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("gemini", config); err == nil {
			c.providers["gemini"] = provider
		}
	}

	// Auto-discover DeepSeek
	if apiKey := os.Getenv("DEEPSEEK_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("deepseek", config); err == nil {
			c.providers["deepseek"] = provider
		}
	}

	// Auto-discover OpenRouter
	if apiKey := os.Getenv("OPENROUTER_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("openrouter", config); err == nil {
			c.providers["openrouter"] = provider
		}
	}

	// Auto-discover Qwen (DashScope)
	if apiKey := os.Getenv("QWEN_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("qwen", config); err == nil {
			c.providers["qwen"] = provider
		}
	}

	// Auto-discover GLM
	if apiKey := os.Getenv("GLM_API_KEY"); apiKey != "" {
		config := ProviderConfig{
			APIKey:     apiKey,
			BaseURL:    getEnvOrDefault("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
			Resilience: c.defaults.Resilience,
		}
		if provider, err := createProvider("glm", config); err == nil {
			c.providers["glm"] = provider
		}
	}
}

// resolveProvider resolves the provider for a model
func (c *Client) resolveProvider(model string) (Provider, error) {
	// Strategy 1: If only one provider is configured, use it directly
	// This allows users to use any model name - let the API validate it
	if len(c.providers) == 1 {
		for _, provider := range c.providers {
			return provider, nil
		}
	}

	// Strategy 2: Multiple providers - must match predefined model lists
	// This ensures correct routing when multiple providers are available
	for _, provider := range c.providers {
		models := provider.Models()
		for _, modelInfo := range models {
			if modelInfo.ID == model {
				return provider, nil
			}
		}
	}

	return nil, fmt.Errorf("no provider found for model '%s'. With multiple providers configured, the model must be in the predefined model list. Available models: %s", model, c.getAvailableModels())
}

// getAvailableModels returns a string of available models for error messages
func (c *Client) getAvailableModels() string {
	var models []string
	for _, provider := range c.providers {
		for _, modelInfo := range provider.Models() {
			models = append(models, modelInfo.ID)
		}
	}
	if len(models) > 10 {
		models = models[:10]
		models = append(models, "...")
	}
	return strings.Join(models, ", ")
}

// applyDefaults applies default configuration to the request
func (c *Client) applyDefaults(req *Request) {
	if req.MaxTokens == nil {
		req.MaxTokens = &c.defaults.MaxTokens
	}
	if req.Temperature == nil {
		req.Temperature = &c.defaults.Temperature
	}
}

// getEnvOrDefault returns environment variable value or default
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Quick performs a quick completion with minimal configuration
// It creates a new client with auto-discovery and makes a simple completion request
func Quick(model, message string) (*Response, error) {
	client := New()
	return client.Chat(context.Background(), &Request{
		Model: model,
		Messages: []Message{
			{Role: "user", Content: message},
		},
	})
}
