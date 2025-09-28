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
		MaxRetries:     3,
		InitialDelay:   1 * time.Second,
		MaxDelay:       30 * time.Second,
		Multiplier:     2.0,
		Jitter:         true,
		RequestTimeout: 30 * time.Second,
		ConnectTimeout: 10 * time.Second,
	}
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

	resilienceConfig := config.Resilience
	if resilienceConfig == (ResilienceConfig{}) {
		resilienceConfig = DefaultResilienceConfig()
	}

	var httpClient HTTPDoer
	if config.HTTPClient != nil {
		httpClient = config.HTTPClient
	} else {
		httpClient = &http.Client{
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
		config.HTTPClient = httpClient
	}

	return &BaseProvider{
		name:             name,
		config:           config,
		httpClient:       httpClient,
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

func getDefaultBaseURL(provider string) string {
	switch provider {
	case "openai":
		return "https://api.openai.com"
	case "anthropic":
		return "https://api.anthropic.com"
	case "gemini":
		return "https://generativelanguage.googleapis.com"
	case "deepseek":
		return "https://api.deepseek.com"
	case "openrouter":
		return "https://openrouter.ai/api/v1"
	case "qwen":
		return "https://dashscope.aliyuncs.com/api/v1"
	case "glm":
		return "https://open.bigmodel.cn/api/paas/v4"
	default:
		return ""
	}
}
