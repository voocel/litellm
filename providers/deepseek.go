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
		IncludeStreamUsage:        true,
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		HasCacheTokens:            true,
	})
}
