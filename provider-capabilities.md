# Provider Capabilities

This page documents the capabilities exposed through the shared `litellm.Request` and `litellm.Response` model. It is based on the current provider adapters, not on every feature a provider may offer in its native API.

Legend: `yes` is supported through the shared API, `no` is not exposed through the shared API, `partial` has provider-specific limits.

Applications can query the same information at runtime:

```go
caps := client.Capabilities(model)
if caps.Thinking.SupportsEffort("high") {
	// Show or enable the "high" thinking option.
}
```

Capability data reports the stable baseline suitable for UI and preflight checks. Provider adapters may encode additional model-specific values; the provider API remains authoritative for those requests.

## Thinking

Portable `Thinking.Effort` values are `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`; accepted values remain model-specific.

| Provider | Enable thinking | Disable thinking | Effort | BudgetTokens | IncludeOutput | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI Chat | partial | partial | low/medium/high/xhigh/max | no | no | Model-specific acceptance is enforced by the OpenAI API. |
| OpenAI Responses | yes | partial | low/medium/high/xhigh/max | no | yes | Also exposes `ReasoningMode`, `ReasoningContext`, and reasoning summaries. |
| Anthropic | yes | unknown | low/medium/high | no | yes | Uses adaptive thinking; disable, `xhigh`, and `max` are model-specific. |
| Bedrock | yes | unknown | low/medium/high | no | no | Claude uses adaptive thinking; disable, `xhigh`, and `max` are model-specific. |
| Gemini | yes | no | high | no | yes | Gemini 3+ uses `thinkingLevel`; other levels are model-specific. |
| DeepSeek | yes | yes | partial | no | no | `low/medium` map to `high`; `xhigh` maps to `max`. |
| GLM | yes | yes | partial | no | no | Model-specific acceptance is enforced by the GLM API. |
| Grok | partial | partial | low/medium/high | no | no | Disable behavior and `xhigh` acceptance are model-specific. |
| OpenRouter | partial | partial | partial | partial | no | Reasoning support depends on the routed model. |
| Ollama | yes | yes | partial | no | no | OpenAI-compatible API accepts `low`, `medium`, and `high`. |
| Qwen | partial | partial | no | partial | no | Thinking mode and budget support are model-specific. |
| MiMo | yes | yes | no | no | no | Thinking is a provider switch; effort and budget controls are rejected. |
| MiniMax | yes | partial | no | no | no | M2 always reasons; M3 supports adaptive or disabled thinking. |

## Reasoning And Usage

| Provider | Reasoning response blocks | Streaming reasoning deltas | Reasoning tokens | Cache read/write usage |
| --- | --- | --- | --- | --- |
| OpenAI Chat | yes | yes | yes | read; write is model-specific |
| OpenAI Responses | yes | yes | yes | read; write is model-specific |
| Anthropic | yes | yes | no | read and write |
| Bedrock | yes | yes | no | read and write |
| Gemini | yes | yes | yes | cache read |
| DeepSeek | yes | yes | yes | cache read |
| GLM | yes | yes | yes | cache read |
| Grok | yes | yes | yes | cache read |
| OpenRouter | yes | yes | yes | read and write |
| Ollama | yes | yes | no | no |
| Qwen | yes | yes | yes | cache read |
| MiMo | yes | yes | yes | cache read |
| MiniMax | yes | yes | yes | cache read |

`ThinkingDisabled` suppresses reasoning output where the provider emits it separately and the adapter can filter it. It does not change provider-native behavior beyond the request fields sent by each adapter.

## Cache Controls

| Provider | Block cache | Request cache policy | Prompt/cache key options |
| --- | --- | --- | --- |
| OpenAI Chat | yes | no | block `Cache` adds an explicit breakpoint; `prompt_cache_key`, `prompt_cache_options`, and legacy `prompt_cache_retention` use provider options |
| OpenAI Responses | yes | no | block `Cache` adds an explicit breakpoint; also supports native `PromptCacheKey`, `PromptCacheOptions`, and legacy `PromptCacheRetention` |
| Anthropic | yes | no | no |
| Bedrock | yes | yes | `cache_retention` provider option |
| Gemini | no | no | no |
| DeepSeek | no | no | no |
| GLM | no | no | no |
| Grok | no | no | no |
| OpenRouter | yes | no | `cache_retention` provider option |
| Ollama | no | no | no |
| Qwen | no | no | no |
| MiMo | no | no | no |
| MiniMax | no | no | no |

## Native APIs

Generic `Client.Chat` and `Client.Stream` use Chat Completions by default. Set `openai.Config.API = openai.APIResponses` to route them through Responses while keeping the shared request and response types. Use `provider/openai.Provider.Responses` and `ResponsesStream` when you need hosted tools, conversation IDs, `previous_response_id`, or other native Responses fields.

Provider-specific request keys are exposed through constants in each provider package; values use `litellm.ProviderOptions` JSON data. Use `litellm.NewProviderOptions` or `Set` to encode them. Unknown provider options are rejected by default; compat providers can opt into pass-through with `AllowUnknownProviderOptions`.

Structured output support follows what the shared adapter can encode. Bedrock exposes JSON schema through `outputConfig.textFormat`; GLM injects the schema into the prompt and sends `json_object`.
