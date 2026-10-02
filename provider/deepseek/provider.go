// Package deepseek connects to the DeepSeek API.
package deepseek

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

// Keys accepted in Request.ProviderOptions, sent as native request fields.
const (
	ProviderOptionLogprobs         = "logprobs"
	ProviderOptionTopLogprobs      = "top_logprobs"
	ProviderOptionUserID           = "user_id"
	ProviderOptionFrequencyPenalty = "frequency_penalty"
	ProviderOptionPresencePenalty  = "presence_penalty"
)

// New connects to https://api.deepseek.com. Thinking maps to thinking.type
// "enabled" or "disabled" with Effort as reasoning_effort; BudgetTokens is
// rejected. Strict tools require BaseURL "https://api.deepseek.com/beta" and
// StrictEnabled on every tool. JSON Schema output uses a prompt with JSON mode;
// schema adherence is not enforced.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:           "deepseek",
		BaseURL:        "https://api.deepseek.com",
		APIKeyRequired: true,
		Thinking:       openaicompat.ThinkingType("enabled", true),
		Options: []string{
			ProviderOptionLogprobs, ProviderOptionTopLogprobs, ProviderOptionUserID,
			ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty,
		},
		ReasoningFields:    []string{"reasoning_content"},
		StringContentRoles: []litellm.Role{litellm.RoleSystem, litellm.RoleAssistant},
		ImageFileID:        true,
		SchemaFallback:     litellm.ResponseFormatJSONObject,
	})
}
