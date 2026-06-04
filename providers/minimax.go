package providers

func init() {
	RegisterBuiltin("minimax", func(cfg ProviderConfig) Provider {
		return NewMiniMax(cfg)
	}, "https://api.minimaxi.com/v1")
}

// NewMiniMax creates a new MiniMax provider using the OpenAI-compatible endpoint.
//
// MiniMax's official API uses a `thinking` object with `type` set to either
// "disabled" or "adaptive" (default: "adaptive").  This is different from
// Qwen/DashScope's `enable_thinking`/`thinking_budget` pattern.
//
// References:
//   - https://platform.minimax.io/docs/api-reference/text-chat-openai
//   - https://platform.minimax.io/docs/api-reference/text-openai-api
func NewMiniMax(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "minimax",
		DefaultBaseURL:            "https://api.minimaxi.com/v1",
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		ThinkingMapper: func(thinking *ThinkingConfig, model string) map[string]any {
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

			return map[string]any{
				"thinking": map[string]any{
					"type": thinkingType,
				},
			}
		},
	})
}
