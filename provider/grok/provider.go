package grok

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.x.ai/v1"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "grok",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			SupportsJSONSchema: true,
			Thinking:           mapThinking,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			HasCompletionTokenDetails: true,
		},
	})
}

func Factory(cfg Config) (litellm.Provider, error) {
	return New(cfg)
}

func mapThinking(thinking *litellm.Thinking, model string) (map[string]any, error) {
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, nil
	}
	if !supportsReasoningEffort(model) {
		return nil, fmt.Errorf("grok: reasoning_effort is only supported by grok-4.3 models, got %q", model)
	}
	if thinking.Mode == litellm.ThinkingDisabled {
		return map[string]any{"reasoning_effort": "none"}, nil
	}
	if thinking.Mode != litellm.ThinkingEnabled {
		return nil, fmt.Errorf("grok: unsupported thinking mode %d", thinking.Mode)
	}
	if thinking.Effort != "" {
		return map[string]any{"reasoning_effort": thinking.Effort}, nil
	}
	if thinking.Level != "" {
		return map[string]any{"reasoning_effort": thinking.Level}, nil
	}
	return nil, fmt.Errorf("grok: thinking level or effort is required")
}

func supportsReasoningEffort(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return model == "grok-4.3" || strings.HasPrefix(model, "grok-4.3-")
}
