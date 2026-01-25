package providers

import (
	"fmt"
	"net"
	"net/http"
	"time"
)

type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// ProviderConfig holds configuration for a provider
type ProviderConfig struct {
	APIKey     string           `json:"api_key"`
	BaseURL    string           `json:"base_url,omitempty"`
	Timeout    time.Duration    `json:"timeout,omitempty"`
	Extra      map[string]any   `json:"extra,omitempty"`
	Resilience ResilienceConfig `json:"resilience,omitempty"`
	HTTPClient HTTPDoer         `json:"-"`
}

// ResilienceConfig holds network resilience configuration for providers
type ResilienceConfig struct {
	MaxRetries     int           `json:"max_retries"`
	InitialDelay   time.Duration `json:"initial_delay"`
	MaxDelay       time.Duration `json:"max_delay"`
	Multiplier     float64       `json:"multiplier"`
	Jitter         bool          `json:"jitter"`
	RequestTimeout time.Duration `json:"request_timeout"`
	ConnectTimeout time.Duration `json:"connect_timeout"`
}

// DefaultResilienceConfig returns default resilience configuration for providers
func DefaultResilienceConfig() ResilienceConfig {
	return ResilienceConfig{
		MaxRetries:     0,
		InitialDelay:   1 * time.Second,
		MaxDelay:       30 * time.Second,
		Multiplier:     2.0,
		Jitter:         true,
		RequestTimeout: 5 * time.Minute,
		ConnectTimeout: 10 * time.Second,
	}
}

// ResolveResilienceConfig applies defaults when config is empty.
func ResolveResilienceConfig(config ResilienceConfig) ResilienceConfig {
	if config == (ResilienceConfig{}) {
		return DefaultResilienceConfig()
	}
	return config
}

// BaseProvider provides common functionality for all providers
type BaseProvider struct {
	name             string
	config           ProviderConfig
	httpClient       HTTPDoer
	resilienceConfig ResilienceConfig
}

// NewBaseProvider creates a new base provider with resilience
func NewBaseProvider(name string, config ProviderConfig) *BaseProvider {
	if config.BaseURL == "" {
		config.BaseURL = getDefaultBaseURL(name)
	}

	resilienceConfig := ResolveResilienceConfig(config.Resilience)

	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{
			Timeout: resilienceConfig.RequestTimeout,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout: resilienceConfig.ConnectTimeout,
				}).DialContext,
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		}
	}

	return &BaseProvider{
		name:             name,
		config:           config,
		httpClient:       config.HTTPClient,
		resilienceConfig: resilienceConfig,
	}
}

func (p *BaseProvider) Name() string {
	return p.name
}

func (p *BaseProvider) Config() ProviderConfig {
	return p.config
}

func (p *BaseProvider) HTTPClient() HTTPDoer {
	return p.httpClient
}

func (p *BaseProvider) ResilienceConfig() ResilienceConfig {
	return p.resilienceConfig
}

func (p *BaseProvider) Validate() error {
	if p.config.APIKey == "" {
		return fmt.Errorf("%s: API key is required", p.name)
	}
	return nil
}

// ValidateExtra validates request-level extra parameters for a provider.
func (p *BaseProvider) ValidateExtra(extra map[string]any, allowedKeys []string) error {
	if len(extra) == 0 {
		return nil
	}
	if len(allowedKeys) == 0 {
		return fmt.Errorf("%s: request extra parameters are not supported", p.name)
	}

	allowed := make(map[string]struct{}, len(allowedKeys))
	for _, key := range allowedKeys {
		allowed[key] = struct{}{}
	}
	for key := range extra {
		if _, ok := allowed[key]; !ok {
			return fmt.Errorf("%s: unsupported extra parameter '%s'", p.name, key)
		}
	}
	return nil
}

// ValidateRequest validates common request parameters
// This should be called by all provider implementations before processing requests
func (p *BaseProvider) ValidateRequest(req *Request) error {
	if req.Model == "" {
		return fmt.Errorf("%s: model is required", p.name)
	}
	if len(req.Messages) == 0 {
		return fmt.Errorf("%s: at least one message is required", p.name)
	}
	if err := validateThinking(req.Thinking); err != nil {
		return fmt.Errorf("%s: %w", p.name, err)
	}

	if req.Temperature != nil {
		if *req.Temperature < 0 || *req.Temperature > 2 {
			return fmt.Errorf("%s: temperature must be between 0 and 2, got %f", p.name, *req.Temperature)
		}
	}

	if req.MaxTokens != nil && *req.MaxTokens <= 0 {
		return fmt.Errorf("%s: max_tokens must be positive, got %d", p.name, *req.MaxTokens)
	}

	return nil
}

func getDefaultBaseURL(provider string) string {
	return GetDefaultURL(provider)
}
