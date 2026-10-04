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
	// DeferredTools reports that the vendor loads the tools marked Deferred
	// once a tool reference names them, the adapter sending every tool from
	// the start; other adapters send what the request offers (see Tool).
	DeferredTools bool
	// ProviderOptions lists, sorted, the Request.ProviderOptions keys the
	// adapter knows: native fields of the vendor's request. One that also
	// passes other keys through, as compat does and the Chat Completions
	// vendors do with AllowUnknownProviderOptions, lists only those it knows.
	ProviderOptions []string
}

// CapabilityProvider is implemented by providers that declare Capabilities.
type CapabilityProvider interface {
	Capabilities() Capabilities
}
