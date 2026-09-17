package minimax

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.minimax.io/v1"

type Config = compat.Config

const ProviderOptionServiceTier = "service_tier"

var allowedProviderOptions = map[string]struct{}{
	ProviderOptionServiceTier: {},
}

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "minimax",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			MaxTokensField:         "max_completion_tokens",
			Thinking:               mapThinking,
			ProviderOptions:        mapProviderOptions,
			AllowedProviderOptions: allowedProviderOptions,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			ReasoningFields:           []string{"reasoning_details", "reasoning_content"},
			HasCompletionTokenDetails: true,
		},
		Stream: compat.StreamSpec{
			ReasoningFields:            []string{"reasoning_details", "reasoning_content"},
			ReasoningCumulative:        true,
			ContentCumulative:          true,
			ContentCumulativeCondition: "thinking_enabled",
		},
		Capabilities: func(model string, caps litellm.Capabilities) litellm.Capabilities {
			caps.Tools.Choice = litellm.SupportPartial
			caps.Thinking.Efforts = nil
			caps.Thinking.BudgetTokens = litellm.SupportNo
			caps.Thinking.IncludeOutput = litellm.SupportNo
			caps.Thinking.Notes = []string{"reasoning_split only separates reasoning output; it does not enable thinking"}
			if isM2(model) {
				caps.Thinking.Supported = litellm.SupportYes
				caps.Thinking.Disable = litellm.SupportNo
				caps.Thinking.Notes = append(caps.Thinking.Notes, "M2.x always reasons and does not accept the M3 thinking parameter")
			} else if isM3(model) {
				caps.Thinking.Supported = litellm.SupportYes
				caps.Thinking.Disable = litellm.SupportYes
				caps.Thinking.Notes = append(caps.Thinking.Notes, "M3 supports adaptive or disabled thinking")
			} else {
				caps.Thinking.Supported = litellm.SupportUnknown
				caps.Thinking.Disable = litellm.SupportUnknown
			}
			return caps
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
	thinkingType := ""
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		if isM2(model) {
			return nil, fmt.Errorf("minimax: thinking cannot be disabled for M2.x models")
		}
		if !isM3(model) {
			return nil, fmt.Errorf("minimax: thinking controls are not supported for %s", model)
		}
		thinkingType = "disabled"
	case litellm.ThinkingEnabled:
		if thinking.Effort != "" {
			return nil, fmt.Errorf("minimax: thinking effort is not supported")
		}
		if thinking.BudgetTokens != nil {
			return nil, fmt.Errorf("minimax: thinking budget_tokens is not supported")
		}
		if isM2(model) {
			return map[string]any{"reasoning_split": true}, nil
		}
		if !isM3(model) {
			return nil, fmt.Errorf("minimax: thinking controls are not supported for %s", model)
		}
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

func mapProviderOptions(options map[string]any, body map[string]any, req *litellm.Request) error {
	if err := validateToolChoice(req.ToolChoice); err != nil {
		return err
	}
	if effectiveThinkingEnabled(req) {
		body["reasoning_split"] = true
	}
	for key, value := range options {
		switch key {
		case ProviderOptionServiceTier:
			tier, ok := value.(string)
			if !ok {
				return fmt.Errorf("minimax: provider option %q must be string", key)
			}
			tier = strings.ToLower(strings.TrimSpace(tier))
			if tier != "standard" && tier != "priority" {
				return fmt.Errorf("minimax: provider option %q must be standard or priority", key)
			}
			body[key] = tier
		default:
			return fmt.Errorf("minimax: unsupported provider option %q", key)
		}
	}
	return nil
}

func validateToolChoice(choice *litellm.ToolChoice) error {
	if choice == nil {
		return nil
	}
	if choice.Name != "" || (choice.Mode != litellm.ToolChoiceAuto && choice.Mode != litellm.ToolChoiceNone) {
		return fmt.Errorf(`minimax: tool_choice only supports "auto" or "none"`)
	}
	return nil
}

func effectiveThinkingEnabled(req *litellm.Request) bool {
	if req.Thinking == nil || req.Thinking.Mode == litellm.ThinkingUnspecified {
		return isM2(req.Model) || isM3(req.Model)
	}
	return req.Thinking.Mode == litellm.ThinkingEnabled
}

func isM2(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "minimax-m2")
}

func isM3(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "minimax-m3")
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
