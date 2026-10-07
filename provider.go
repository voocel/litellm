package litellm

import (
	"context"
	"net/http"
	"time"
)

// Provider adapts one vendor protocol. Name identifies it in responses,
// errors and ProviderState, and must not be empty. Chat and Stream receive a
// validated copy of the request that the provider owns. Implementations may
// also implement CapabilityProvider.
type Provider interface {
	Name() string
	Chat(context.Context, *Request) (*Response, error)
	Stream(context.Context, *Request) (Stream, error)
}

// HTTPClient sends provider requests; *http.Client satisfies it.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// ModelLister is implemented by providers that list the models their
// credentials reach, which also checks the credentials: a rejected key fails
// with ErrorTypeAuth.
type ModelLister interface {
	ListModels(context.Context) ([]ModelInfo, error)
}

// ModelInfo is a model as the vendor lists it, in the vendor's order.
type ModelInfo struct {
	ID string
	// Name is the vendor's display name; empty when it gives none.
	Name string
	// Created is when the vendor released or listed the model; zero when it
	// does not say.
	Created time.Time
}
