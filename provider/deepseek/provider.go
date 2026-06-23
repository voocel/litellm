package deepseek

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.deepseek.com"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	supportsStrict := strings.HasSuffix(strings.TrimRight(cfg.BaseURL, "/"), "/beta")
	strictTools := compat.StrictToolsOmit
	if supportsStrict {
		strictTools = compat.StrictToolsRequireAll
	}
	return compat.New(cfg, compat.Spec{
		Name: "deepseek",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			Thinking: mapThinking,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			ReasoningFields:           []string{"reasoning_content"},
			HasCompletionTokenDetails: true,
			HasCacheTokens:            true,
		},
		Stream: compat.StreamSpec{
			ReasoningFields: []string{"reasoning_content"},
		},
		Features: compat.FeatureSpec{
			StrictTools: strictTools,
		},
	})
}

func Factory(cfg Config) (litellm.Provider, error) {
	return New(cfg)
}

func mapThinking(thinking *litellm.Thinking, _ string) (map[string]any, error) {
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, nil
	}
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		return map[string]any{"thinking": map[string]any{"type": "disabled"}}, nil
	case litellm.ThinkingEnabled:
		body := map[string]any{"thinking": map[string]any{"type": "enabled"}}
		if effort := thinkingEffort(thinking.Level); effort != "" {
			body["reasoning_effort"] = effort
		}
		return body, nil
	default:
		return nil, fmt.Errorf("deepseek: unsupported thinking mode %d", thinking.Mode)
	}
}

func thinkingEffort(level string) string {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "":
		return ""
	case "low", "medium", "high":
		return "high"
	case "xhigh", "max":
		return "max"
	default:
		return strings.TrimSpace(level)
	}
}
