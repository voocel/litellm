// Package claude holds the Claude request mapping shared by the Anthropic and
// Bedrock adapters.
package claude

import "github.com/voocel/litellm"

// ThinkingConfig is Claude's thinking object; Effort travels separately in
// output_config.
type ThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
	Effort       string `json:"-"`
}

// Thinking maps t: a budget selects "enabled" with budget_tokens, otherwise
// "adaptive". Values are sent as given. It returns nil when t is nil.
func Thinking(t *litellm.Thinking) *ThinkingConfig {
	if t == nil {
		return nil
	}
	if t.Mode == litellm.ThinkingDisabled {
		return &ThinkingConfig{Type: "disabled"}
	}
	out := &ThinkingConfig{Type: "adaptive", Effort: t.Effort}
	if t.BudgetTokens != nil {
		out.Type, out.BudgetTokens = "enabled", t.BudgetTokens
	}
	if t.IncludeOutput {
		out.Display = "summarized"
	}
	return out
}
