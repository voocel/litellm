package qwen

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

type Config = compat.Config

const (
	ProviderOptionTopK                   = "top_k"
	ProviderOptionRepetitionPenalty      = "repetition_penalty"
	ProviderOptionPresencePenalty        = "presence_penalty"
	ProviderOptionVLHighResolutionImages = "vl_high_resolution_images"
	ProviderOptionN                      = "n"
	ProviderOptionModalities             = "modalities"
	ProviderOptionAudio                  = "audio"
	ProviderOptionPreserveThinking       = "preserve_thinking"
	ProviderOptionToolStream             = "tool_stream"
	ProviderOptionEnableCodeInterpreter  = "enable_code_interpreter"
	ProviderOptionSeed                   = "seed"
	ProviderOptionLogprobs               = "logprobs"
	ProviderOptionTopLogprobs            = "top_logprobs"
	ProviderOptionParallelToolCalls      = "parallel_tool_calls"
	ProviderOptionEnableSearch           = "enable_search"
	ProviderOptionSearchOptions          = "search_options"
	ProviderOptionSkill                  = "skill"
)

var allowedProviderOptions = map[string]struct{}{
	ProviderOptionTopK:                   {},
	ProviderOptionRepetitionPenalty:      {},
	ProviderOptionPresencePenalty:        {},
	ProviderOptionVLHighResolutionImages: {},
	ProviderOptionN:                      {},
	ProviderOptionModalities:             {},
	ProviderOptionAudio:                  {},
	ProviderOptionPreserveThinking:       {},
	ProviderOptionToolStream:             {},
	ProviderOptionEnableCodeInterpreter:  {},
	ProviderOptionSeed:                   {},
	ProviderOptionLogprobs:               {},
	ProviderOptionTopLogprobs:            {},
	ProviderOptionParallelToolCalls:      {},
	ProviderOptionEnableSearch:           {},
	ProviderOptionSearchOptions:          {},
	ProviderOptionSkill:                  {},
}

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "qwen",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			MaxTokensField:         "max_completion_tokens",
			Thinking:               mapThinking,
			AllowedProviderOptions: allowedProviderOptions,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			ReasoningFields:           []string{"reasoning_content"},
			HasCompletionTokenDetails: true,
		},
		Stream: compat.StreamSpec{
			ReasoningFields: []string{"reasoning_content"},
		},
		Capabilities: func(_ string, caps litellm.Capabilities) litellm.Capabilities {
			caps.Thinking.Supported = litellm.SupportPartial
			caps.Thinking.Disable = litellm.SupportPartial
			caps.Thinking.Efforts = nil
			caps.Thinking.BudgetTokens = litellm.SupportPartial
			caps.Thinking.IncludeOutput = litellm.SupportNo
			caps.Thinking.Notes = []string{"thinking mode and budget support are model-specific; Effort is rejected"}
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
	if thinking.Mode == litellm.ThinkingDisabled {
		return map[string]any{"enable_thinking": false}, nil
	}
	if thinking.Mode != litellm.ThinkingEnabled {
		return nil, fmt.Errorf("qwen: unsupported thinking mode %d", thinking.Mode)
	}
	if thinking.Effort != "" {
		return nil, fmt.Errorf("qwen: thinking effort is not supported; use budget_tokens")
	}
	body := map[string]any{}
	if !thinkingOnly(model) {
		body["enable_thinking"] = true
	}
	if thinking.BudgetTokens != nil {
		if *thinking.BudgetTokens <= 0 {
			return nil, fmt.Errorf("qwen: thinking budget_tokens must be positive")
		}
		body["thinking_budget"] = *thinking.BudgetTokens
	}
	return body, nil
}

func thinkingOnly(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	if strings.Contains(model, "-thinking") || strings.HasPrefix(model, "qwq-") || strings.HasPrefix(model, "qvq-") {
		return true
	}
	switch model {
	case "qwen3.7-max-preview", "qwen3.7-max-2026-05-17":
		return true
	default:
		return false
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
