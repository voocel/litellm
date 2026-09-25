package litellm

import (
	"context"
	"net/http"
)

// Provider adapts one vendor protocol. Chat and Stream receive a validated
// copy of the request that the provider owns. Implementations may also
// implement ModelLister and CapabilityProvider.
type Provider interface {
	Name() string
	Chat(context.Context, *Request) (*Response, error)
	Stream(context.Context, *Request) (Stream, error)
}

// HTTPClient sends provider requests; *http.Client satisfies it.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// ModelLister is implemented by providers that can list models.
type ModelLister interface {
	ListModels(context.Context) ([]ModelInfo, error)
}

// ModelInfo describes a model; fields the vendor does not report are zero.
type ModelInfo struct {
	ID               string
	Name             string
	Provider         string
	Description      string
	ContextLength    int
	InputTokenLimit  int
	OutputTokenLimit int
	Created          int64
}
