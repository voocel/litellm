package providers

import (
	"fmt"
	"strings"
)

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
		ThinkingMapper: mapGLMThinking,
	})
}

// mapGLMThinking translates the unified thinking config into GLM's request params:
//
//   - thinking: {type: enabled|disabled} — supported since GLM-4.5.
//   - reasoning_effort: minimal|low|medium|high|xhigh|... — supported only since
//     GLM-5.2 (per docs.bigmodel.cn). GLM accepts the level strings verbatim, so
//     no remapping is needed.
//
// Both are version-gated because older GLM models reject params they don't know:
// below GLM-4.5 nothing is sent (thinking itself is unsupported); from GLM-4.5 up
// to 5.1 only the on/off switch is sent and the level is dropped.
func mapGLMThinking(thinking *ThinkingConfig, model string) map[string]any {
	if thinking == nil || !glmVersionAtLeast(model, 4, 5) {
		return nil
	}
	if isThinkingDisabledConfig(thinking) {
		return map[string]any{"thinking": map[string]any{"type": "disabled"}}
	}
	if !isThinkingEnabledConfig(thinking) {
		return nil
	}
	body := map[string]any{"thinking": map[string]any{"type": "enabled"}}
	if thinking.Level != "" && glmVersionAtLeast(model, 5, 2) {
		body["reasoning_effort"] = strings.ToLower(strings.TrimSpace(thinking.Level))
	}
	return body
}

// glmVersionAtLeast parses the major.minor version from a GLM model name and
// reports whether it is at least major.minor. Unparseable / non-GLM names return
// false so older or custom-named models aren't sent params they would reject.
func glmVersionAtLeast(model string, major, minor int) bool {
	_, after, ok := strings.Cut(strings.ToLower(model), "glm-")
	if !ok {
		return false
	}
	var maj, min int
	fmt.Sscanf(after, "%d.%d", &maj, &min)
	return maj > major || (maj == major && min >= minor)
}
