package litellm

import "context"

// Option is a functional option for configuring the client
type Option func(*Client)

// ProviderConfig contains provider-specific configuration
type ProviderConfig struct {
	APIKey  string `json:"api_key"`
	BaseURL string `json:"base_url,omitempty"`

	// Resilience configuration integrated directly
	Resilience ResilienceConfig `json:"resilience,omitempty"`

	// Provider-specific extras
	Extra map[string]any `json:"extra,omitempty"`
}

// Context keys for request metadata
type contextKey string

const (
	ContextKeyRequestID  contextKey = "request_id"
	ContextKeyRetryCount contextKey = "retry_count"
	ContextKeyProvider   contextKey = "provider"
)

// ChatProvider defines the basic chat completion capability
type ChatProvider interface {
	Chat(ctx context.Context, req *Request) (*Response, error)
}

// StreamProvider defines streaming capability
type StreamProvider interface {
	Stream(ctx context.Context, req *Request) (StreamReader, error)
}

// ModelProvider defines model information capability
type ModelProvider interface {
	Models() []ModelInfo
	SupportsModel(model string) bool
}

// Provider combines all capabilities through interface composition
// Implementations can choose which interfaces to support
type Provider interface {
	ChatProvider
	StreamProvider
	ModelProvider

	// Basic provider info
	Name() string
	Validate() error
}

// ProviderFactory is a function that creates a provider instance
type ProviderFactory func(config ProviderConfig) Provider
