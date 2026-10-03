// Package ollama connects to Ollama's OpenAI-compatible endpoint.
package ollama

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
	ProviderOptionSeed             = "seed"
)

// New sends Thinking.Effort as reasoning_effort and Thinking.Disabled as
// reasoning_effort "none". No API key is required.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:    "ollama",
		BaseURL: "http://localhost:11434/v1",
		Options: []string{
			ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty, ProviderOptionSeed,
		},
		ReasoningFields: []string{"reasoning", "reasoning_content", "thinking"},
	})
}
