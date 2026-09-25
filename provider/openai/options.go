package openai

import "slices"

// ProviderOptions are native request fields of the selected API, copied into
// the body as is. An option naming a generated object or array (such as
// Responses "text", "reasoning" or "tools") is merged into or appended to it.
const (
	ProviderOptionFrequencyPenalty     = "frequency_penalty"
	ProviderOptionPresencePenalty      = "presence_penalty"
	ProviderOptionLogitBias            = "logit_bias"
	ProviderOptionN                    = "n"
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
	ProviderOptionModalities           = "modalities"
	ProviderOptionAudio                = "audio"
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
	ProviderOptionPreviousResponseID = "previous_response_id"
	ProviderOptionConversation       = "conversation"
	ProviderOptionInclude            = "include"
	ProviderOptionTruncation         = "truncation"
	ProviderOptionMaxToolCalls       = "max_tool_calls"
	ProviderOptionBackground         = "background"
	ProviderOptionContextManagement  = "context_management"
	ProviderOptionPrompt             = "prompt"
	// ProviderOptionText is merged into the generated text object, e.g.
	// {"verbosity": "low"}.
	ProviderOptionText = "text"
	// ProviderOptionReasoning is merged into the generated reasoning object,
	// e.g. {"summary": "auto"}.
	ProviderOptionReasoning = "reasoning"
	// ProviderOptionTools is appended to the generated tools, e.g. hosted
	// tools such as [{"type": "web_search"}].
	ProviderOptionTools = "tools"
)

var chatOptions = []string{
	ProviderOptionFrequencyPenalty, ProviderOptionPresencePenalty, ProviderOptionLogitBias,
	ProviderOptionN, ProviderOptionLogprobs, ProviderOptionTopLogprobs, ProviderOptionStore,
	ProviderOptionModeration, ProviderOptionStreamOptions, ProviderOptionPromptCacheKey,
	ProviderOptionPromptCacheOptions, ProviderOptionPromptCacheRetention, ProviderOptionPrediction,
	ProviderOptionMetadata, ProviderOptionModalities, ProviderOptionAudio, ProviderOptionServiceTier,
	ProviderOptionSafetyIdentifier, ProviderOptionUser, ProviderOptionVerbosity,
	ProviderOptionWebSearchOptions, ProviderOptionParallelToolCalls, ProviderOptionSeed,
}

var responsesOptions = []string{
	ProviderOptionStore, ProviderOptionStreamOptions, ProviderOptionPromptCacheKey,
	ProviderOptionPromptCacheOptions, ProviderOptionPromptCacheRetention, ProviderOptionMetadata,
	ProviderOptionServiceTier, ProviderOptionSafetyIdentifier, ProviderOptionParallelToolCalls,
	ProviderOptionTopLogprobs, ProviderOptionPreviousResponseID, ProviderOptionConversation,
	ProviderOptionInclude, ProviderOptionTruncation, ProviderOptionMaxToolCalls,
	ProviderOptionBackground, ProviderOptionContextManagement, ProviderOptionPrompt,
	ProviderOptionText, ProviderOptionReasoning, ProviderOptionTools,
}

func sortedCopy(keys []string) []string {
	out := slices.Clone(keys)
	slices.Sort(out)
	return out
}
