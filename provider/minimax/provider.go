// Package minimax connects to the MiniMax API.
package minimax

import (
	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

// Config and Provider are shared with package compat.
type (
	Config   = compat.Config
	Provider = compat.Provider
)

// ProviderOptionServiceTier is accepted in Request.ProviderOptions.
const ProviderOptionServiceTier = "service_tier"

// New always sends reasoning_split so reasoning arrives in reasoning_details
// instead of <think> tags inside content. Thinking maps to thinking.type
// "adaptive" or "disabled"; Effort and BudgetTokens are rejected.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:            "minimax",
		BaseURL:         "https://api.minimax.io/v1",
		APIKeyRequired:  true,
		MaxTokensField:  "max_completion_tokens",
		Thinking:        openaicompat.ThinkingType("adaptive", false),
		Fields:          map[string]any{"reasoning_split": true},
		Options:         []string{ProviderOptionServiceTier},
		ReasoningFields: []string{"reasoning_details", "reasoning_content"},
		SchemaFallback:  litellm.ResponseFormatText,
	})
}
