package litellm

// Capabilities states what a provider adapter can express on the wire. It is
// static per provider: whether a model honors a request is the vendor's call,
// and the vendor's error is the source of truth.
type Capabilities struct {
	// Thinking reports that Request.Thinking is sent to the vendor.
	Thinking bool
	// DisableThinking, ThinkingEffort and ThinkingBudget report that
	// ThinkingDisabled, Thinking.Effort and Thinking.BudgetTokens are sent;
	// otherwise the adapter rejects them before sending.
	DisableThinking bool
	ThinkingEffort  bool
	ThinkingBudget  bool
	// MaxTokensRequired reports that the vendor rejects requests without
	// Request.MaxTokens.
	MaxTokensRequired bool
	// ProviderOptions lists the accepted Request.ProviderOptions keys, sorted.
	// A provider that passes every key through, such as compat, lists none.
	ProviderOptions []string
}

// CapabilityProvider is implemented by providers that declare Capabilities.
type CapabilityProvider interface {
	Capabilities() Capabilities
}
