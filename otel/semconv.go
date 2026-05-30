package otel

// GenAI semantic-convention attribute keys recorded on generation spans.
//
// Based on the OpenTelemetry GenAI semantic conventions
// (https://opentelemetry.io/docs/specs/semconv/gen-ai/). The official Go
// semconv module has not stabilized this group yet, so the keys are declared
// here. Langfuse (and other OTLP backends) treat any span carrying
// gen_ai.request.model as an LLM generation and map these attributes into their
// data model.
const (
	attrSystem          = "gen_ai.system"
	attrRequestModel    = "gen_ai.request.model"
	attrResponseModel   = "gen_ai.response.model"
	attrFinishReason    = "gen_ai.response.finish_reason"
	attrInputTokens     = "gen_ai.usage.input_tokens"
	attrOutputTokens    = "gen_ai.usage.output_tokens"
	attrCacheReadTokens = "gen_ai.usage.cache_read_input_tokens"
	attrPrompt          = "gen_ai.prompt"
	attrCompletion      = "gen_ai.completion"
)
