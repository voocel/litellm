// Package mimo connects to the Xiaomi MiMo API.
package mimo

import (
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
	ProviderOptionAudio            = "audio"
	ProviderOptionFrequencyPenalty = "frequency_penalty"
	ProviderOptionPresencePenalty  = "presence_penalty"
)

// New connects to https://api.xiaomimimo.com/v1. Thinking maps to
// thinking.type "enabled" or "disabled"; Effort and BudgetTokens are
// rejected.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:              "mimo",
		BaseURL:           "https://api.xiaomimimo.com/v1",
		APIKeyRequired:    true,
		MaxTokensField:    "max_completion_tokens",
		Thinking:          openaicompat.ThinkingType("enabled", false),
		Options:           []string{ProviderOptionAudio, ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty},
		ReasoningFields:   []string{"reasoning_content"},
		OmitStreamOptions: true,
	})
}
