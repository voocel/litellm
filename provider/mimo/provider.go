package mimo

import (
	"fmt"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.xiaomimimo.com/v1"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "mimo",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			Thinking: mapThinking,
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

func mapThinking(thinking *litellm.Thinking, _ string) (map[string]any, error) {
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, nil
	}
	kwargs := map[string]any{}
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		kwargs["enable_thinking"] = false
	case litellm.ThinkingEnabled:
		kwargs["enable_thinking"] = true
	default:
		return nil, fmt.Errorf("mimo: unsupported thinking mode %d", thinking.Mode)
	}
	return map[string]any{"chat_template_kwargs": kwargs}, nil
}
