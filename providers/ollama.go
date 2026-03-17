package providers

func init() {
	RegisterBuiltin("ollama", func(cfg ProviderConfig) Provider {
		return NewOllama(cfg)
	}, "http://localhost:11434/v1")
}

// NewOllama creates a new Ollama provider using the OpenAI-compatible endpoint.
func NewOllama(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:         "ollama",
		DefaultBaseURL:       "http://localhost:11434/v1",
		ModelFromResponse:    true,
		SkipAPIKeyValidation: true,
	})
}
