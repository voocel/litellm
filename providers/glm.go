package providers

func init() {
	RegisterBuiltin("glm", func(cfg ProviderConfig) Provider {
		return NewGLM(cfg)
	}, "https://open.bigmodel.cn/api/paas/v4")
}

// NewGLM creates a new GLM/ZhiPu AI provider.
func NewGLM(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "glm",
		DefaultBaseURL:            "https://open.bigmodel.cn/api/paas/v4",
		MaxStopSequences:          1,
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		JSONSchemaToPrompt:        true,
		ResponseFormatMapper: func(rf *ResponseFormat) any {
			if rf.Type == "json_object" || rf.Type == "json_schema" {
				return map[string]string{"type": "json_object"}
			}
			return nil
		},
	})
}
