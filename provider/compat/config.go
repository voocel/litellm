// Package compat connects to any OpenAI-compatible Chat Completions endpoint,
// such as vLLM, LM Studio or a gateway, and holds the Config shared by the
// vendor wrappers (deepseek, glm, grok, mimo, minimax, ollama, openrouter,
// qwen).
package compat

import "github.com/voocel/litellm/provider/internal/openaicompat"

type (
	// Config holds the connection settings; its fields are documented on the
	// aliased type.
	Config = openaicompat.Config
	// Provider implements litellm.Provider and litellm.CapabilityProvider.
	Provider = openaicompat.Provider
)

// New speaks the plain Chat Completions protocol: max_tokens, reasoning_effort
// and no vendor extensions. BaseURL is required and the API key optional.
// Every ProviderOption is copied into the request body as is.
func New(cfg Config) (*Provider, error) {
	cfg.AllowUnknownProviderOptions = true
	return openaicompat.New(cfg, openaicompat.Spec{
		Name:            "compat",
		ReasoningFields: []string{"reasoning_content", "reasoning"},
	})
}
