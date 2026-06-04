package providers

import "strings"

func init() {
	RegisterBuiltin("deepseek", func(cfg ProviderConfig) Provider {
		return NewDeepSeek(cfg)
	}, "https://api.deepseek.com")
	RegisterBuiltin("deepseek-anthropic", func(cfg ProviderConfig) Provider {
		return NewDeepSeekAnthropic(cfg)
	}, "https://api.deepseek.com/anthropic")
}

// NewDeepSeek creates a new DeepSeek provider.
func NewDeepSeek(config ProviderConfig) *OpenAICompatProvider {
	supportsStrict := strings.HasSuffix(strings.TrimSuffix(config.BaseURL, "/"), "/beta")
	return NewOpenAICompat(config, Compat{
		ProviderName:              "deepseek",
		DefaultBaseURL:            "https://api.deepseek.com",
		ModelFromResponse:         true,
		HasCompletionTokenDetails: true,
		HasCacheTokens:            true,
		// DeepSeek supports tools[i].function.strict on its beta endpoint.
		// Callers should set BaseURL to https://api.deepseek.com/beta for it.
		// Its beta strict mode requires every function tool to set strict=true.
		SupportsStrictTools:   supportsStrict,
		RequireAllToolsStrict: true,
		ThinkingMapper:        mapDeepSeekThinking,
	})
}

// NewDeepSeekAnthropic creates a DeepSeek provider using its Anthropic-compatible API.
func NewDeepSeekAnthropic(config ProviderConfig) *AnthropicProvider {
	return newAnthropicProvider("deepseek-anthropic", config, true)
}

func mapDeepSeekThinking(thinking *ThinkingConfig, _ string) map[string]any {
	if thinking == nil {
		return nil
	}

	config := map[string]any{}
	if isThinkingDisabledConfig(thinking) {
		config["type"] = "disabled"
		return map[string]any{"thinking": config}
	}
	if !isThinkingEnabledConfig(thinking) {
		return nil
	}

	config["type"] = "enabled"
	body := map[string]any{"thinking": config}
	if effort := deepSeekThinkingEffort(thinking.Level); effort != "" {
		body["reasoning_effort"] = effort
	}
	return body
}

func deepSeekThinkingEffort(level string) string {
	level = strings.TrimSpace(level)
	switch strings.ToLower(level) {
	case "":
		return ""
	case "low", "medium", "high":
		return "high"
	case "xhigh", "max":
		return "max"
	default:
		return level
	}
}

func resolveDeepSeekAnthropicThinking(req *Request) *ThinkingConfig {
	thinking := normalizeThinking(req)
	if thinking == nil {
		return nil
	}
	return &ThinkingConfig{Type: thinking.Type}
}

func deepSeekThinkingEffortFromRequest(req *Request) string {
	thinking := normalizeThinking(req)
	if thinking == nil || !isThinkingEnabledConfig(thinking) {
		return ""
	}
	return deepSeekThinkingEffort(thinking.Level)
}
