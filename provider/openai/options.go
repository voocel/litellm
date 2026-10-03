package openai

import "slices"

// ProviderOptions are native request fields of the selected API, copied into
// the body as is. An option naming a generated object (such as Responses
// "text" or "reasoning") is merged into it. Responses options that add output
// litellm does not model, hosted tools and server-side compaction, are not
// offered, as their items would drop out of the conversation; nor is
// previous_response_id, as a Response carries no id. Results reported beside
// the output, such as moderation scores, are in Response.Raw when captured.
const (
	ProviderOptionFrequencyPenalty     = "frequency_penalty"
	ProviderOptionPresencePenalty      = "presence_penalty"
	ProviderOptionLogitBias            = "logit_bias"
	ProviderOptionLogprobs             = "logprobs"
	ProviderOptionTopLogprobs          = "top_logprobs"
	ProviderOptionStore                = "store"
	ProviderOptionModeration           = "moderation"
	ProviderOptionStreamOptions        = "stream_options"
	ProviderOptionPromptCacheKey       = "prompt_cache_key"
	ProviderOptionPromptCacheOptions   = "prompt_cache_options"
	ProviderOptionPromptCacheRetention = "prompt_cache_retention"
	ProviderOptionPrediction           = "prediction"
	ProviderOptionMetadata             = "metadata"
	ProviderOptionServiceTier          = "service_tier"
	ProviderOptionSafetyIdentifier     = "safety_identifier"
	ProviderOptionUser                 = "user"
	ProviderOptionVerbosity            = "verbosity"
	ProviderOptionWebSearchOptions     = "web_search_options"
	ProviderOptionParallelToolCalls    = "parallel_tool_calls"
	ProviderOptionSeed                 = "seed"
)

// Responses API options (Config.API = APIResponses) not listed above.
const (
	ProviderOptionConversation = "conversation"
	ProviderOptionInclude      = "include"
	ProviderOptionTruncation   = "truncation"
	// ProviderOptionBackground is supported by Stream only; Chat has no job
	// polling interface and rejects background=true.
	ProviderOptionBackground = "background"
	ProviderOptionPrompt     = "prompt"
	// ProviderOptionText is merged into the generated text object, e.g.
	// {"verbosity": "low"}.
	ProviderOptionText = "text"
	// ProviderOptionReasoning is merged into the generated reasoning object,
	// e.g. {"summary": "auto"}.
	ProviderOptionReasoning = "reasoning"
)

var chatOptions = []string{
	ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty, ProviderOptionLogitBias,
	ProviderOptionLogprobs, ProviderOptionTopLogprobs, ProviderOptionStore,
	ProviderOptionModeration, ProviderOptionStreamOptions, ProviderOptionPromptCacheKey,
	ProviderOptionPromptCacheOptions, ProviderOptionPromptCacheRetention, ProviderOptionPrediction,
	ProviderOptionMetadata, ProviderOptionServiceTier,
	ProviderOptionSafetyIdentifier, ProviderOptionUser, ProviderOptionVerbosity,
	ProviderOptionWebSearchOptions, ProviderOptionParallelToolCalls, ProviderOptionSeed,
}

var responsesOptions = []string{
	ProviderOptionStore, ProviderOptionModeration, ProviderOptionStreamOptions, ProviderOptionPromptCacheKey,
	ProviderOptionPromptCacheOptions, ProviderOptionPromptCacheRetention, ProviderOptionMetadata,
	ProviderOptionServiceTier, ProviderOptionSafetyIdentifier, ProviderOptionUser, ProviderOptionParallelToolCalls,
	ProviderOptionTopLogprobs, ProviderOptionConversation, ProviderOptionInclude,
	ProviderOptionTruncation, ProviderOptionBackground, ProviderOptionPrompt,
	ProviderOptionText, ProviderOptionReasoning,
}

func sortedCopy(keys []string) []string {
	out := slices.Clone(keys)
	slices.Sort(out)
	return out
}
