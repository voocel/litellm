package litellm

import (
	"context"
	"net/http"
)

// Provider adapts one vendor protocol. Chat and Stream receive a validated
// copy of the request that the provider owns. Implementations may also
// implement CapabilityProvider.
type Provider interface {
	Name() string
	Chat(context.Context, *Request) (*Response, error)
	Stream(context.Context, *Request) (Stream, error)
}

// HTTPClient sends provider requests; *http.Client satisfies it.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}
