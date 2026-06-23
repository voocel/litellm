package qwen

import (
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "qwen",
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
		},
		Stream: compat.StreamSpec{
			ReasoningFields: []string{"reasoning_content"},
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
		return nil, fmt.Errorf("qwen: disabling thinking is not supported by the OpenAI-compatible endpoint")
	}
	if thinking.Mode != litellm.ThinkingEnabled {
		return nil, fmt.Errorf("qwen: unsupported thinking mode %d", thinking.Mode)
	}
	body := map[string]any{"enable_thinking": true}
	if thinking.BudgetTokens != nil {
		body["thinking_budget"] = *thinking.BudgetTokens
	}
	return body, nil
}
