package providers

func init() {
	RegisterBuiltin("mimo", func(cfg ProviderConfig) Provider {
		return NewMimo(cfg)
	}, "https://api.xiaomimimo.com/v1")
}

// NewMimo creates a new Xiaomi MiMo provider using the OpenAI-compatible
// endpoint.
//
// MiMo (v2.5 / v2.5-pro / v2-pro / v2-omni / v2-flash) is reasoning-aware.
// Thinking is gated by a non-standard nested field on the request body:
//
//	"chat_template_kwargs": { "enable_thinking": true|false }
//
// Without this field, MiMo's vLLM/SGLang servers fall back to the model's
// chat-template default (v2.5-pro defaults to enabled), but reasoning content
// is *not streamed* incrementally — it gets buffered until the thinking phase
// completes, leaving the SSE stream silent for the duration. That silence
// trips the per-chunk idle watchdog on long-context, long-output workloads.
// Forwarding enable_thinking explicitly keeps the server in streaming
// reasoning mode (delta.reasoning_content / delta.reasoning, both probed by
// the default reasoning field list).
//
// Reasoning text is delivered through the standard probe list — no
// ReasoningField override needed (vLLM emits "reasoning_content" on older
// builds, "reasoning" on newer ones; both are covered).
//
// References:
//   - vLLM Recipe (V2.5-Pro): https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5-Pro
//   - HuggingFace model card:  https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
//   - Open platform docs:      https://platform.xiaomimimo.com/
func NewMimo(config ProviderConfig) *OpenAICompatProvider {
	return NewOpenAICompat(config, Compat{
		ProviderName:              "mimo",
		DefaultBaseURL:            "https://api.xiaomimimo.com/v1",
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		ThinkingMapper: func(thinking *ThinkingConfig, _ string) map[string]any {
			// nil → server defaults (v2.5-pro: enabled; older: disabled).
			// Returning nil keeps the request portable and avoids overriding
			// a deployment that knows what it's doing.
			if thinking == nil {
				return nil
			}
			kwargs := map[string]any{}
			switch {
			case isThinkingDisabledConfig(thinking):
				kwargs["enable_thinking"] = false
			case isThinkingEnabledConfig(thinking):
				kwargs["enable_thinking"] = true
			default:
				return nil
			}
			return map[string]any{"chat_template_kwargs": kwargs}
		},
	})
}
