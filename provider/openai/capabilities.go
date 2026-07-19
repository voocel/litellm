package openai

import (
	"net/url"
	"strings"

	"github.com/voocel/litellm"
)

// promptCacheParamsSupport reports whether this endpoint is trusted to accept
// OpenAI's prompt cache params (prompt_cache_key / prompt_cache_retention).
// Only the official endpoint guarantees the field contract; see
// Config.PromptCacheParams for the opt-in on compatible backends.
func (p *Provider) promptCacheParamsSupport() litellm.Support {
	if p.cfg.PromptCacheParams || isOfficialBaseURL(p.cfg.BaseURL) {
		return litellm.SupportYes
	}
	return litellm.SupportUnknown
}

func isOfficialBaseURL(baseURL string) bool {
	u, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	return strings.EqualFold(u.Hostname(), "api.openai.com")
}

// structuredSupport reports the endpoint contract implemented by this
// provider. The official OpenAI Chat/Responses APIs accept Structured Outputs;
// a compatible custom endpoint makes no such guarantee and remains Unknown.
// Model-specific exceptions belong to endpoint/model metadata or an explicit
// caller override, not an ever-growing model-name list here.
func (p *Provider) structuredSupport() litellm.StructuredCapabilities {
	if !isOfficialBaseURL(p.cfg.BaseURL) {
		return litellm.StructuredCapabilities{
			JSONObject: litellm.SupportUnknown,
			JSONSchema: litellm.SupportUnknown,
			Strict:     litellm.SupportUnknown,
		}
	}
	return litellm.StructuredCapabilities{
		JSONObject: litellm.SupportYes,
		JSONSchema: litellm.SupportYes,
		Strict:     litellm.SupportYes,
	}
}

func (p *Provider) Capabilities(model string) litellm.Capabilities {
	reasoningModel := p.isReasoningModel(model)
	thinking := litellm.ThinkingCapabilities{
		Supported: litellm.SupportPartial,
		Disable:   litellm.SupportYes,
		Efforts:   []string{"low", "medium", "high", "xhigh"},
		Notes:     []string{"chat reasoning controls are only available on reasoning chat models"},
	}
	if !reasoningModel {
		thinking.Supported = litellm.SupportNo
		thinking.Efforts = nil
	}
	return litellm.Capabilities{
		Provider: p.Name(),
		Model:    model,
		Thinking: thinking,
		Reasoning: litellm.ReasoningCapabilities{
			Blocks:          litellm.SupportYes,
			StreamingDeltas: litellm.SupportYes,
			ReasoningTokens: litellm.SupportYes,
		},
		Tools: litellm.ToolCapabilities{
			Calls:               litellm.SupportYes,
			ParallelCalls:       litellm.SupportYes,
			StrictSchema:        litellm.SupportYes,
			Choice:              litellm.SupportYes,
			HostedProviderTools: litellm.SupportPartial,
		},
		Structured: p.structuredSupport(),
		Media: litellm.MediaCapabilities{
			ImageURL:    litellm.SupportYes,
			ImageBytes:  litellm.SupportYes,
			FileURI:     litellm.SupportNo,
			ImageDetail: litellm.SupportYes,
		},
		Cache: litellm.CacheCapabilities{
			Block:      litellm.SupportNo,
			PromptKey:  p.promptCacheParamsSupport(),
			Retention:  p.promptCacheParamsSupport(),
			UsageRead:  litellm.SupportYes,
			UsageWrite: litellm.SupportNo,
		},
		Streaming: litellm.StreamingCapabilities{
			Supported:       litellm.SupportYes,
			Usage:           litellm.SupportYes,
			ReasoningDeltas: litellm.SupportYes,
			ToolCallDeltas:  litellm.SupportYes,
			NativeResponses: litellm.SupportYes,
			IdleTimeout:     litellm.SupportYes,
		},
		Usage: litellm.UsageCapabilities{
			InputTokens:      litellm.SupportYes,
			OutputTokens:     litellm.SupportYes,
			TotalTokens:      litellm.SupportYes,
			ReasoningTokens:  litellm.SupportYes,
			CacheReadTokens:  litellm.SupportYes,
			CacheWriteTokens: litellm.SupportNo,
		},
	}
}
