package litellm

// Capabilities states protocol facts of a provider adapter. It is static per
// provider: whether a model honors a request is the vendor's call, and the
// vendor's error is the source of truth.
type Capabilities struct {
	// MaxTokensRequired reports that the vendor rejects requests without
	// Request.MaxTokens.
	MaxTokensRequired bool
	// ThinkingEffort and DisableThinking report that the adapter sends
	// Thinking.Effort and Thinking.Disabled; it rejects them otherwise, before
	// the request is sent. Which efforts a model takes is the vendor's call.
	ThinkingEffort  bool
	DisableThinking bool
	// ProviderOptions lists the accepted Request.ProviderOptions keys, sorted.
	// A provider that passes every key through, such as compat, lists none.
	ProviderOptions []string
}

// CapabilityProvider is implemented by providers that declare Capabilities.
type CapabilityProvider interface {
	Capabilities() Capabilities
}
