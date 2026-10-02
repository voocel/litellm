// Package qwen connects to the Alibaba Cloud DashScope compatible-mode API.
package qwen

import (
	"errors"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

// Config and Provider are shared with package compat.
type (
	Config   = compat.Config
	Provider = compat.Provider
)

// Keys accepted in Request.ProviderOptions, sent as native request fields.
const (
	ProviderOptionTopK                   = "top_k"
	ProviderOptionRepetitionPenalty      = "repetition_penalty"
	ProviderOptionPresencePenalty        = "presence_penalty"
	ProviderOptionVLHighResolutionImages = "vl_high_resolution_images"
	ProviderOptionPreserveThinking       = "preserve_thinking"
	ProviderOptionToolStream             = "tool_stream"
	ProviderOptionSeed                   = "seed"
	ProviderOptionLogprobs               = "logprobs"
	ProviderOptionTopLogprobs            = "top_logprobs"
	ProviderOptionParallelToolCalls      = "parallel_tool_calls"
	ProviderOptionEnableSearch           = "enable_search"
	ProviderOptionSearchOptions          = "search_options"
	ProviderOptionSkill                  = "skill"
)

// New connects to https://dashscope.aliyuncs.com/compatible-mode/v1; set
// Config.BaseURL for other regions.
// Thinking maps to enable_thinking, with BudgetTokens as thinking_budget;
// Effort is rejected.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:           "qwen",
		BaseURL:        "https://dashscope.aliyuncs.com/compatible-mode/v1",
		APIKeyRequired: true,
		MaxTokensField: "max_completion_tokens",
		Thinking:       mapThinking,
		Options: []string{
			ProviderOptionTopK, ProviderOptionRepetitionPenalty, ProviderOptionPresencePenalty,
			ProviderOptionVLHighResolutionImages, ProviderOptionPreserveThinking, ProviderOptionToolStream,
			ProviderOptionSeed, ProviderOptionLogprobs,
			ProviderOptionTopLogprobs, ProviderOptionParallelToolCalls, ProviderOptionEnableSearch,
			ProviderOptionSearchOptions, ProviderOptionSkill,
		},
		ReasoningFields: []string{"reasoning_content"},
	})
}

// mapThinking uses DashScope's enable_thinking switch and thinking_budget.
func mapThinking(thinking *litellm.Thinking) (map[string]any, error) {
	if thinking.Disabled {
		return map[string]any{"enable_thinking": false}, nil
	}
	if thinking.Effort != "" {
		return nil, errors.New("thinking effort is not supported; use budget_tokens")
	}
	body := map[string]any{"enable_thinking": true}
	if thinking.BudgetTokens != nil {
		body["thinking_budget"] = *thinking.BudgetTokens
	}
	return body, nil
}
