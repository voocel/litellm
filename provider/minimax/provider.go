package minimax

import (
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.minimax.io/v1"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "minimax",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			MaxTokensField: "max_completion_tokens",
			Thinking:       mapThinking,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			ReasoningFields:           []string{"reasoning_details"},
			HasCompletionTokenDetails: true,
		},
		Stream: compat.StreamSpec{
			ReasoningFields:     []string{"reasoning_details", "reasoning_content"},
			ReasoningCumulative: true,
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
	thinkingType := ""
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		thinkingType = "disabled"
	case litellm.ThinkingEnabled:
		thinkingType = "adaptive"
	default:
		return nil, fmt.Errorf("minimax: unsupported thinking mode %d", thinking.Mode)
	}
	body := map[string]any{"thinking": map[string]any{"type": thinkingType}}
	if thinkingType == "adaptive" {
		body["reasoning_split"] = true
	}
	return body, nil
}
