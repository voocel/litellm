package providers

func init() {
	RegisterBuiltin("grok", func(cfg ProviderConfig) Provider {
		return NewGrok(cfg)
	}, "https://api.x.ai/v1")
}

// NewGrok creates a new Grok (xAI) provider.
func NewGrok(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "grok",
		DefaultBaseURL:            "https://api.x.ai/v1",
		IncludeStreamUsage:        true,
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		SupportsJSONSchema:        true,
	})
}
