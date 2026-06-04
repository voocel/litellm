package providers

func init() {
	RegisterBuiltin("minimax", func(cfg ProviderConfig) Provider {
		return NewMiniMax(cfg)
	}, "https://api.minimax.io/v1")
}

// NewMiniMax creates a new MiniMax provider using the OpenAI-compatible endpoint.
//
// The default endpoint follows MiniMax's international OpenAI-compatible API.
// China-region users can override ProviderConfig.BaseURL with
// "https://api.minimaxi.com/v1".
//
// MiniMax's official API uses a `thinking` object with `type` set to either
// "disabled" or "adaptive" (default: "adaptive").  When thinking is enabled,
// `reasoning_split` asks MiniMax to return reasoning separately instead of
// embedding it in the content as <think> tags.
//
// References:
//   - https://platform.minimax.io/docs/api-reference/text-chat-openai
//   - https://platform.minimax.io/docs/api-reference/text-openai-api
func NewMiniMax(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "minimax",
		DefaultBaseURL:            "https://api.minimax.io/v1",
		MaxTokensField:            "max_completion_tokens",
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		ReasoningField:            "reasoning_details",
		ReasoningCumulative:       true,
		ThinkingMapper: func(thinking *ThinkingConfig, _ string) map[string]any {
			if thinking == nil {
				return nil
			}

			// Map the generic "enabled" → MiniMax "adaptive"
			var thinkingType string
			switch {
			case isThinkingEnabledConfig(thinking):
				thinkingType = "adaptive"
			case isThinkingDisabledConfig(thinking):
				thinkingType = "disabled"
			default:
				return nil
			}

			body := map[string]any{
				"thinking": map[string]any{
					"type": thinkingType,
				},
			}
			if thinkingType == "adaptive" {
				body["reasoning_split"] = true
			}
			return body
		},
	})
}
