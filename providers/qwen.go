package providers

func init() {
	RegisterBuiltin("qwen", func(cfg ProviderConfig) Provider {
		return NewQwen(cfg)
	}, "https://dashscope.aliyuncs.com/compatible-mode/v1")
}

// NewQwen creates a new Qwen provider using the OpenAI-compatible endpoint.
func NewQwen(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "qwen",
		DefaultBaseURL:            "https://dashscope.aliyuncs.com/compatible-mode/v1",
		ModelFromResponse:         true,
		IncludeStreamUsage:        true,
		HasCompletionTokenDetails: true,
		ThinkingMapper: func(thinking *ThinkingConfig, model string) map[string]any {
			if thinking == nil || thinking.Type != "enabled" {
				return nil
			}
			m := map[string]any{"enable_thinking": true}
			if budget := ResolveBudgetTokens(thinking); budget != nil {
				m["thinking_budget"] = *budget
			}
			return m
		},
	})
}
