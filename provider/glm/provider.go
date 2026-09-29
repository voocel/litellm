// Package glm connects to the Zhipu GLM API.
package glm

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
	ProviderOptionDoSample  = "do_sample"
	ProviderOptionRequestID = "request_id"
	// ProviderOptionThinking is merged into the thinking object built from
	// Request.Thinking, e.g. {"clear_thinking": false}.
	ProviderOptionThinking   = "thinking"
	ProviderOptionToolStream = "tool_stream"
	ProviderOptionUserID     = "user_id"
)

// New connects to https://open.bigmodel.cn/api/paas/v4. Thinking maps to
// thinking.type "enabled" or "disabled" with Effort as reasoning_effort;
// BudgetTokens is rejected.
func New(cfg Config) (*Provider, error) {
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:           "glm",
		BaseURL:        "https://open.bigmodel.cn/api/paas/v4",
		APIKeyRequired: true,
		Thinking:       openaicompat.ThinkingType("enabled", true),
		Options: []string{
			ProviderOptionDoSample, ProviderOptionRequestID, ProviderOptionThinking,
			ProviderOptionToolStream, ProviderOptionUserID,
		},
		ReasoningFields: []string{"reasoning_content"},
		SchemaFallback:  litellm.ResponseFormatJSONObject,
		// Cached prompt tokens bill at a discount; caching itself is free.
		CacheWritesUnbilled: true,
	})
}
