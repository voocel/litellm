package mimo

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://api.xiaomimimo.com/v1"

type Config = compat.Config

const (
	ProviderOptionAudio            = "audio"
	ProviderOptionFrequencyPenalty = "frequency_penalty"
	ProviderOptionPresencePenalty  = "presence_penalty"
)

var allowedProviderOptions = map[string]struct{}{
	ProviderOptionAudio:            {},
	ProviderOptionFrequencyPenalty: {},
	ProviderOptionPresencePenalty:  {},
}

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "mimo",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			MaxStopSequences:       4,
			MaxTokensField:         "max_completion_tokens",
			Thinking:               mapThinking,
			ProviderOptions:        mapProviderOptions,
			AllowedProviderOptions: allowedProviderOptions,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			ReasoningFields:           []string{"reasoning_content"},
			HasCompletionTokenDetails: true,
		},
		Stream: compat.StreamSpec{
			ReasoningFields:   []string{"reasoning_content"},
			OmitStreamOptions: true,
		},
		Features: compat.FeatureSpec{
			StrictTools: compat.StrictToolsForward,
		},
		Capabilities: func(model string, caps litellm.Capabilities) litellm.Capabilities {
			caps.Tools.Choice = litellm.SupportPartial
			if thinkingUnsupported(model) {
				caps.Thinking.Supported = litellm.SupportNo
				caps.Thinking.Disable = litellm.SupportNo
			}
			caps.Thinking.Efforts = nil
			caps.Thinking.BudgetTokens = litellm.SupportNo
			caps.Thinking.IncludeOutput = litellm.SupportNo
			caps.Thinking.Notes = []string{"thinking is a provider switch; effort and budget controls are rejected"}
			caps.Streaming.Usage = litellm.SupportNo
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
	if thinkingUnsupported(model) {
		return nil, fmt.Errorf("mimo: thinking is not supported for %s", model)
	}
	thinkingType := ""
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		thinkingType = "disabled"
	case litellm.ThinkingEnabled:
		if thinking.Effort != "" {
			return nil, fmt.Errorf("mimo: thinking effort is not supported")
		}
		if thinking.BudgetTokens != nil {
			return nil, fmt.Errorf("mimo: thinking budget_tokens is not supported")
		}
		thinkingType = "enabled"
	default:
		return nil, fmt.Errorf("mimo: unsupported thinking mode %d", thinking.Mode)
	}
	return map[string]any{"thinking": map[string]any{"type": thinkingType}}, nil
}

func mapProviderOptions(options map[string]any, body map[string]any, req *litellm.Request) error {
	if effectiveThinkingEnabled(req.Thinking, req.Model) && thinkingOverridesSampling(req.Model) {
		if req.Temperature != nil {
			return fmt.Errorf("mimo: temperature cannot be customized when thinking is enabled for %s", req.Model)
		}
		if req.TopP != nil {
			return fmt.Errorf("mimo: top_p cannot be customized when thinking is enabled for %s", req.Model)
		}
	}
	if err := validateToolChoice(req.ToolChoice); err != nil {
		return err
	}
	for key, value := range options {
		switch key {
		case ProviderOptionAudio:
			if _, ok := value.(map[string]any); !ok {
				return fmt.Errorf("mimo: provider option %q must be object", key)
			}
			body[key] = value
		case ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty:
			switch value.(type) {
			case json.Number, float64, float32, int, int64, int32, nil:
				body[key] = value
			default:
				return fmt.Errorf("mimo: provider option %q must be number or null", key)
			}
		default:
			return fmt.Errorf("mimo: unsupported provider option %q", key)
		}
	}
	return nil
}

func validateToolChoice(choice *litellm.ToolChoice) error {
	if choice == nil {
		return nil
	}
	if choice.Name != "" || choice.Mode != litellm.ToolChoiceAuto {
		return fmt.Errorf(`mimo: tool_choice only supports "auto"`)
	}
	return nil
}

func effectiveThinkingEnabled(thinking *litellm.Thinking, model string) bool {
	if thinking != nil {
		switch thinking.Mode {
		case litellm.ThinkingEnabled:
			return true
		case litellm.ThinkingDisabled:
			return false
		}
	}
	return thinkingDefaultEnabled(model)
}

func thinkingDefaultEnabled(model string) bool {
	return usesCurrentThinkingContract(model)
}

func thinkingOverridesSampling(model string) bool {
	return usesCurrentThinkingContract(model)
}

func thinkingUnsupported(model string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(model)), "-tts")
}

func usesCurrentThinkingContract(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	if thinkingUnsupported(model) || !strings.HasPrefix(model, "mimo-v") {
		return false
	}
	version := strings.TrimPrefix(model, "mimo-v")
	version, _, _ = strings.Cut(version, "-")
	var major, minor int
	if _, err := fmt.Sscanf(version, "%d.%d", &major, &minor); err != nil {
		return false
	}
	return major > 2 || major == 2 && minor >= 5
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
