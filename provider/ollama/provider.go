package ollama

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "http://localhost:11434/v1"

type Config = compat.Config

const (
	ProviderOptionFrequencyPenalty = "frequency_penalty"
	ProviderOptionPresencePenalty  = "presence_penalty"
	ProviderOptionSeed             = "seed"
	ProviderOptionLogitBias        = "logit_bias"
	ProviderOptionUser             = "user"
	ProviderOptionN                = "n"
)

var allowedProviderOptions = map[string]struct{}{
	ProviderOptionFrequencyPenalty: {},
	ProviderOptionPresencePenalty:  {},
	ProviderOptionSeed:             {},
	ProviderOptionLogitBias:        {},
	ProviderOptionUser:             {},
	ProviderOptionN:                {},
}

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "ollama",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Request: compat.RequestSpec{
			Thinking:               mapThinking,
			AllowedProviderOptions: allowedProviderOptions,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse: true,
			ReasoningFields:   []string{"reasoning", "reasoning_content", "thinking"},
		},
		Stream: compat.StreamSpec{
			ReasoningFields: []string{"reasoning", "reasoning_content", "thinking"},
		},
		Capabilities: func(_ string, caps litellm.Capabilities) litellm.Capabilities {
			caps.Thinking.Efforts = []string{"low", "medium", "high"}
			caps.Thinking.BudgetTokens = litellm.SupportNo
			caps.Thinking.IncludeOutput = litellm.SupportNo
			caps.Thinking.Notes = []string{"OpenAI-compatible reasoning_effort accepts low, medium, high, or none"}
			caps.Reasoning.ReasoningTokens = litellm.SupportNo
			caps.Usage.ReasoningTokens = litellm.SupportNo
			caps.Usage.CacheReadTokens = litellm.SupportNo
			caps.Usage.CacheWriteTokens = litellm.SupportNo
			return caps
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
	return map[string]any{"reasoning_effort": "high"}, nil
}

func reasoningEffort(effort string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	switch normalized {
	case "high", "medium", "low":
		return normalized, nil
	default:
		return "", fmt.Errorf("ollama: unsupported reasoning effort %q; use low, medium, or high", effort)
	}
}

// NewClient builds the provider from cfg and wraps it in a ready *litellm.Client.
// It is a convenience for the common single-provider case. It calls New(cfg)
// and then litellm.New(provider, opts...).
func NewClient(cfg Config, opts ...litellm.ClientOption) (*litellm.Client, error) {
	p, err := New(cfg)
	if err != nil {
		return nil, err
	}
	return litellm.New(p, opts...)
}
