package providers

func init() {
	RegisterBuiltin("deepseek", func(cfg ProviderConfig) Provider {
		return NewDeepSeek(cfg)
	}, "https://api.deepseek.com")
}

// NewDeepSeek creates a new DeepSeek provider.
func NewDeepSeek(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "deepseek",
		DefaultBaseURL:            "https://api.deepseek.com",
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		HasCacheTokens:            true,
		// DeepSeek supports tools[i].function.strict on its beta endpoint.
		// Callers should set BaseURL to https://api.deepseek.com/beta for it.
		SupportsStrictTools: true,
	})
}
