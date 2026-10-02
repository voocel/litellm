// Package openrouter connects to the OpenRouter API.
package openrouter

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
	// ProviderOptionCacheControl is the top-level cache_control object that
	// enables automatic prompt caching, e.g. {"type": "ephemeral"}.
	ProviderOptionCacheControl = "cache_control"
	ProviderOptionSessionID    = "session_id"
	// ProviderOptionRouting is OpenRouter's provider routing object.
	ProviderOptionRouting = "provider"
)

// New connects to https://openrouter.ai/api/v1. Thinking maps to the reasoning
// object (effort or max_tokens, which OpenRouter accepts one at a time, or
// effort "none" when disabled) and CacheControl to cache_control on content
// parts.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:            "openrouter",
		BaseURL:         "https://openrouter.ai/api/v1",
		APIKeyRequired:  true,
		Thinking:        mapThinking,
		Options:         []string{ProviderOptionCacheControl, ProviderOptionSessionID, ProviderOptionRouting},
		Cache:           map[string]any{"cache_control": map[string]any{"type": "ephemeral"}},
		ReasoningFields: []string{"reasoning_details", "reasoning", "reasoning_content"},
	})
}

// mapThinking fills OpenRouter's unified reasoning object.
func mapThinking(thinking *litellm.Thinking) (map[string]any, error) {
	if thinking.Disabled {
		return map[string]any{"reasoning": map[string]any{"effort": "none"}}, nil
	}
	if thinking.Effort != "" && thinking.BudgetTokens != nil {
		return nil, errors.New("thinking effort and budget_tokens cannot be combined")
	}
	reasoning := map[string]any{}
	if thinking.Effort != "" {
		reasoning["effort"] = thinking.Effort
	}
	if thinking.BudgetTokens != nil {
		reasoning["max_tokens"] = *thinking.BudgetTokens
	}
	if len(reasoning) == 0 {
		reasoning["enabled"] = true
	}
	return map[string]any{"reasoning": reasoning}, nil
}
