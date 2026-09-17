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
			ProviderOptions:    mapProviderOptions,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			HasCompletionTokenDetails: true,
		},
		Features: compat.FeatureSpec{StrictTools: compat.StrictToolsAlways},
		Capabilities: func(_ string, caps litellm.Capabilities) litellm.Capabilities {
			caps.Thinking.Supported = litellm.SupportPartial
			caps.Thinking.Disable = litellm.SupportPartial
			caps.Thinking.Efforts = []string{"low", "medium", "high"}
			caps.Thinking.BudgetTokens = litellm.SupportNo
			caps.Thinking.IncludeOutput = litellm.SupportNo
			caps.Thinking.Notes = []string{"reasoning support, disable behavior, and xhigh acceptance are model-specific"}
			caps.Structured.JSONSchema = litellm.SupportYes
			caps.Structured.Strict = litellm.SupportYes
			return caps
		},
	})
}

func Factory(cfg Config) (litellm.Provider, error) {
	return New(cfg)
}

func mapProviderOptions(options map[string]any, body map[string]any, req *litellm.Request) error {
	reasoningEnabled := req.Thinking != nil && req.Thinking.Mode == litellm.ThinkingEnabled
	if reasoningEnabled && len(req.Stop) > 0 {
		return fmt.Errorf("grok: stop is not supported for reasoning models")
	}
	for key, value := range options {
		if reasoningEnabled && isUnsupportedReasoningOption(key) {
			return fmt.Errorf("grok: provider option %q is not supported for reasoning models", key)
		}
		if _, exists := body[key]; exists {
			return fmt.Errorf("grok: provider option %q conflicts with generated request field", key)
		}
		body[key] = value
	}
	return nil
}

func mapThinking(thinking *litellm.Thinking, _ string) (map[string]any, error) {
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, nil
	}
	if thinking.Mode == litellm.ThinkingDisabled {
		return map[string]any{"reasoning_effort": "none"}, nil
	}
	if thinking.Mode != litellm.ThinkingEnabled {
		return nil, fmt.Errorf("grok: unsupported thinking mode %d", thinking.Mode)
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

func isUnsupportedReasoningOption(key string) bool {
	switch key {
	case "stop", "presence_penalty", "frequency_penalty", "presencePenalty", "frequencyPenalty":
		return true
	default:
		return false
	}
}

func reasoningEffort(effort string) (string, error) {
	normalized := strings.ToLower(strings.TrimSpace(effort))
	switch normalized {
	case "low", "medium", "high", "xhigh":
		return normalized, nil
	default:
		return "", fmt.Errorf("grok: unsupported reasoning_effort %q; use low, medium, high, or xhigh", effort)
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
