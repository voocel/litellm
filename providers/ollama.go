package providers

import "strings"

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
		ThinkingMapper:       mapOllamaThinking,
	})
}

// mapOllamaThinking translates the unified thinking config into Ollama's
// OpenAI-compatible reasoning_effort field (Ollama converts it to its native
// Think param). Ollama only accepts "low"/"medium"/"high"/"none" on the /v1
// endpoint — "minimal" and "xhigh" are rejected — so levels are clamped:
// minimal→low, xhigh→high. "none" disables thinking, which is the only way to
// stop Ollama from silently auto-enabling reasoning on capable models.
//
// Note: Ollama's OpenAI compatibility is experimental and model-dependent;
// reasoning_effort only takes effect on thinking-capable models (qwen3,
// deepseek-r1, …). Picking a non-reasoning model with a thinking level set is
// the caller's responsibility, as with the other OpenAI-compatible providers.
func mapOllamaThinking(thinking *ThinkingConfig, _ string) map[string]any {
	if thinking == nil {
		return nil
	}
	if isThinkingDisabledConfig(thinking) {
		return map[string]any{"reasoning_effort": "none"}
	}
	if !isThinkingEnabledConfig(thinking) {
		return nil
	}
	if effort := ollamaReasoningEffort(thinking.Level); effort != "" {
		return map[string]any{"reasoning_effort": effort}
	}
	return nil // enabled but no level: let Ollama auto-think at its default
}

func ollamaReasoningEffort(level string) string {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "":
		return ""
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high", "xhigh":
		return "high"
	default:
		return level
	}
}
