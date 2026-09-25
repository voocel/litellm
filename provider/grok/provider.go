// Package grok connects to the xAI Grok API.
package grok

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
	ProviderOptionFrequencyPenalty = "frequency_penalty"
	ProviderOptionPresencePenalty  = "presence_penalty"
	ProviderOptionLogprobs         = "logprobs"
	ProviderOptionTopLogprobs      = "top_logprobs"
	ProviderOptionN                = "n"
	ProviderOptionUser             = "user"
)

// New sends Thinking.Effort as reasoning_effort. xAI reasoning models cannot
// disable thinking, so ThinkingDisabled is rejected.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:             "grok",
		BaseURL:          "https://api.x.ai/v1",
		APIKeyRequired:   true,
		ThinkingAlwaysOn: true,
		Options: []string{
			ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty, ProviderOptionLogprobs,
			ProviderOptionTopLogprobs, ProviderOptionN, ProviderOptionUser,
		},
		ReasoningFields: []string{"reasoning_content"},
	})
}
