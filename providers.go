package litellm

import (
	"fmt"
	"net/http"
	"time"
)

// ProviderRegistry holds all registered provider factories
var ProviderRegistry = make(map[string]ProviderFactory)

// ProviderFactory creates a provider instance
type ProviderFactory func(config ProviderConfig) Provider

// RegisterProvider registers a provider factory
func RegisterProvider(name string, factory ProviderFactory) {
	ProviderRegistry[name] = factory
}

// CreateProvider creates a provider instance by name
func CreateProvider(name string, config ProviderConfig) (Provider, error) {
	factory, exists := ProviderRegistry[name]
	if !exists {
		return nil, fmt.Errorf("provider '%s' not found", name)
	}
	return factory(config), nil
}

// BaseProvider provides common functionality for all providers
type BaseProvider struct {
	name       string
	config     ProviderConfig
	httpClient *http.Client
}

// NewBaseProvider creates a new base provider
func NewBaseProvider(name string, config ProviderConfig) *BaseProvider {
	if config.BaseURL == "" {
		config.BaseURL = getDefaultBaseURL(name)
	}

	return &BaseProvider{
		name:   name,
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Name returns the provider name
func (p *BaseProvider) Name() string {
	return p.name
}

// Config returns the provider configuration
func (p *BaseProvider) Config() ProviderConfig {
	return p.config
}

// HTTPClient returns the HTTP client
func (p *BaseProvider) HTTPClient() *http.Client {
	return p.httpClient
}

// Validate checks if the provider is properly configured
func (p *BaseProvider) Validate() error {
	if p.config.APIKey == "" {
		return fmt.Errorf("%s: API key is required", p.name)
	}
	return nil
}

// getDefaultBaseURL returns the default base URL for a provider
func getDefaultBaseURL(provider string) string {
	switch provider {
	case "openai":
		return "https://api.openai.com"
	case "anthropic":
		return "https://api.anthropic.com"
	case "gemini":
		return "https://generativelanguage.googleapis.com"
	default:
		return ""
	}
}

// Factory functions for built-in providers
func createOpenAIProvider(config ProviderConfig) Provider {
	if factory, exists := ProviderRegistry["openai"]; exists {
		return factory(config)
	}
	return nil
}

func createAnthropicProvider(config ProviderConfig) Provider {
	if factory, exists := ProviderRegistry["anthropic"]; exists {
		return factory(config)
	}
	return nil
}

func createGeminiProvider(config ProviderConfig) Provider {
	if factory, exists := ProviderRegistry["gemini"]; exists {
		return factory(config)
	}
	return nil
}
