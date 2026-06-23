package ollama

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "http://localhost:11434/v1"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "ollama",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Request: compat.RequestSpec{
			Thinking: mapThinking,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse: true,
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
	if thinking.Mode == litellm.ThinkingDisabled {
		return map[string]any{"reasoning_effort": "none"}, nil
	}
	if thinking.Mode != litellm.ThinkingEnabled {
		return nil, fmt.Errorf("ollama: unsupported thinking mode %d", thinking.Mode)
	}
	if thinking.Effort != "" {
		return map[string]any{"reasoning_effort": thinking.Effort}, nil
	}
	if effort := reasoningEffort(thinking.Level); effort != "" {
		return map[string]any{"reasoning_effort": effort}, nil
	}
	return nil, fmt.Errorf("ollama: thinking level or effort is required")
}

func reasoningEffort(level string) string {
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
		return strings.TrimSpace(level)
	}
}
