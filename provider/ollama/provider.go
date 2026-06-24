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
			ReasoningFields:   []string{"reasoning", "reasoning_content", "thinking"},
		},
		Stream: compat.StreamSpec{
			ReasoningFields: []string{"reasoning", "reasoning_content", "thinking"},
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
		effort, err := reasoningEffort(thinking.Effort)
		if err != nil {
			return nil, err
		}
		return map[string]any{"reasoning_effort": effort}, nil
	}
	if thinking.Level != "" {
		effort, err := reasoningLevel(thinking.Level)
		if err != nil {
			return nil, err
		}
		return map[string]any{"reasoning_effort": effort}, nil
	}
	return nil, fmt.Errorf("ollama: thinking level or effort is required")
}

func reasoningEffort(effort string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	switch normalized {
	case "high", "medium", "low", "max", "none":
		return normalized, nil
	default:
		return "", fmt.Errorf("ollama: unsupported reasoning effort %q", effort)
	}
}

func reasoningLevel(level string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "minimal", "low":
		return "low", nil
	case "medium":
		return "medium", nil
	case "high":
		return "high", nil
	case "xhigh", "max":
		return "max", nil
	default:
		return "", fmt.Errorf("ollama: unsupported thinking level %q", level)
	}
}
