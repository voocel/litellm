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
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		SupportsJSONSchema:        true,
		// Grok uses top-level reasoning_effort ("low"/"medium"/"high").
		// Only sent when Level is explicitly set; models default to their own behavior.
		ThinkingMapper: func(thinking *ThinkingConfig, _ string) map[string]any {
			if thinking.Type != "enabled" || thinking.Level == "" {
				return nil
			}
			return map[string]any{"reasoning_effort": thinking.Level}
		},
	})
}
