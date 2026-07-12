package anthropic

import "github.com/voocel/litellm"

func (p *Provider) Capabilities(model string) litellm.Capabilities {
	caps := litellm.Capabilities{
		Provider: p.Name(),
		Model:    model,
		Thinking: litellm.ThinkingCapabilities{
			Supported: litellm.SupportYes,
			Disable:   litellm.SupportYes,
			Efforts:   litellm.PortableThinkingEfforts(),
		},
		Reasoning: litellm.ReasoningCapabilities{
			Blocks:          litellm.SupportYes,
			StreamingDeltas: litellm.SupportYes,
			ReasoningTokens: litellm.SupportNo,
		},
		Tools: litellm.ToolCapabilities{
			Calls:               litellm.SupportYes,
			StrictSchema:        litellm.SupportYes,
			Choice:              litellm.SupportPartial,
			MultimodalResults:   litellm.SupportYes,
			RoundTripSignatures: litellm.SupportYes,
		},
		Structured: litellm.StructuredCapabilities{
			JSONObject: litellm.SupportNo,
			JSONSchema: litellm.SupportYes,
			Strict:     litellm.SupportYes,
		},
		Media: litellm.MediaCapabilities{
			ImageURL:   litellm.SupportYes,
			ImageBytes: litellm.SupportYes,
			FileURI:    litellm.SupportNo,
		},
		Cache: litellm.CacheCapabilities{
			Block:      litellm.SupportYes,
			Retention:  litellm.SupportYes,
			UsageRead:  litellm.SupportYes,
			UsageWrite: litellm.SupportYes,
		},
		Streaming: litellm.StreamingCapabilities{
			Supported:       litellm.SupportYes,
			Usage:           litellm.SupportYes,
			ReasoningDeltas: litellm.SupportYes,
			ToolCallDeltas:  litellm.SupportYes,
			IdleTimeout:     litellm.SupportYes,
		},
		Usage: litellm.UsageCapabilities{
			InputTokens:      litellm.SupportYes,
			OutputTokens:     litellm.SupportYes,
			TotalTokens:      litellm.SupportYes,
			ReasoningTokens:  litellm.SupportNo,
			CacheReadTokens:  litellm.SupportYes,
			CacheWriteTokens: litellm.SupportYes,
		},
	}
	switch classifyModel(model) {
	case familyAlwaysThinking:
		caps.Thinking.Disable = litellm.SupportNo
		caps.Thinking.BudgetTokens = litellm.SupportNo
		caps.Thinking.IncludeOutput = litellm.SupportYes
		caps.Thinking.Notes = []string{
			"thinking is always on; disabled requests omit the thinking field with a warning",
			"effort maps to output_config.effort; minimal maps to low",
		}
	case familyAdaptive:
		caps.Thinking.BudgetTokens = litellm.SupportNo
		caps.Thinking.IncludeOutput = litellm.SupportYes
		caps.Thinking.Notes = []string{
			"adaptive thinking; effort maps to output_config.effort; minimal maps to low",
			"budget_tokens is dropped with a warning",
		}
	case familyClaude46:
		caps.Thinking.BudgetTokens = litellm.SupportPartial
		caps.Thinking.IncludeOutput = litellm.SupportNo
		caps.Thinking.Notes = []string{
			"adaptive thinking unless budget_tokens is set (deprecated escape hatch)",
			"effort maps to output_config.effort; minimal maps to low, xhigh maps to max",
		}
	default:
		caps.Thinking.BudgetTokens = litellm.SupportYes
		caps.Thinking.IncludeOutput = litellm.SupportNo
		caps.Thinking.Notes = []string{
			"extended thinking; effort maps to budget_tokens and requires max_tokens headroom",
		}
	}
	return caps
}
