package providers

import "context"

// StreamReader reads streaming chunks from a provider.
type StreamReader interface {
	Next() (*StreamChunk, error)
	Close() error
}

// Provider is the core interface that all LLM providers implement.
type Provider interface {
	Name() string
	Validate() error
	Chat(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (StreamReader, error)
}

// ModelLister is an optional interface implemented by providers that support listing models.
type ModelLister interface {
	ListModels(ctx context.Context) ([]ModelInfo, error)
}
