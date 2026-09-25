package otel

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/voocel/litellm"
)

// GenAI semantic-convention attribute keys recorded on generation spans.
//
// Based on the OpenTelemetry GenAI semantic conventions
// (https://github.com/open-telemetry/semantic-conventions-genai). The official
// Go semconv module has not stabilized this group yet, so the keys are kept
// local to this optional module.
const (
	attrProviderName     = "gen_ai.provider.name"
	attrOperationName    = "gen_ai.operation.name"
	attrRequestModel     = "gen_ai.request.model"
	attrRequestStream    = "gen_ai.request.stream"
	attrResponseModel    = "gen_ai.response.model"
	attrFinishReasons    = "gen_ai.response.finish_reasons"
	attrInputTokens      = "gen_ai.usage.input_tokens"
	attrOutputTokens     = "gen_ai.usage.output_tokens"
	attrReasoningTokens  = "gen_ai.usage.reasoning.output_tokens"
	attrCacheReadTokens  = "gen_ai.usage.cache_read.input_tokens"
	attrCacheWriteTokens = "gen_ai.usage.cache_creation.input_tokens"
	attrInputMessages    = "gen_ai.input.messages"
	attrOutputMessages   = "gen_ai.output.messages"
	attrErrorType        = "error.type"
)

// semanticOperation names the operation: Gemini's API is generateContent, the
// others are chat APIs.
func semanticOperation(provider string) string {
	if provider == "gemini" {
		return "generate_content"
	}
	return "chat"
}

func semanticProvider(provider string) string {
	switch provider {
	case "bedrock":
		return "aws.bedrock"
	case "gemini":
		return "gcp.gemini"
	case "grok":
		return "x_ai"
	default:
		return provider
	}
}

func semanticFinishReason(reason litellm.FinishReason) string {
	switch reason {
	case litellm.FinishReasonToolCall:
		return "tool_call"
	case litellm.FinishReasonSafety:
		return "content_filter"
	case "":
		return "unknown"
	default:
		return string(reason)
	}
}

func semanticErrorType(err error) string {
	if err == nil {
		return ""
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "timeout"
	}
	if errors.Is(err, context.Canceled) {
		return "canceled"
	}
	var llmErr *litellm.Error
	if errors.As(err, &llmErr) {
		if llmErr.Code != "" {
			return llmErr.Code
		}
		if llmErr.StatusCode != 0 {
			return strconv.Itoa(llmErr.StatusCode)
		}
		if llmErr.Type != "" {
			return string(llmErr.Type)
		}
	}
	return fmt.Sprintf("%T", err)
}
