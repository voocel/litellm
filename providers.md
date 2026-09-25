# Provider Mapping

Each adapter maps the shared `litellm.Request` onto its vendor's wire format and nothing more. It does not infer what a model supports and does not check vendor values; whatever is sent is judged by the vendor API, and its error is returned as is. This page records the mapping so you can predict the request an adapter sends.

`client.Capabilities()` reports the same static facts at runtime: whether `Thinking`, `ThinkingDisabled`, `Effort` and `BudgetTokens` can be expressed (the table below), and the accepted `ProviderOptions` keys.

## Output Limit

| Provider | `MaxTokens` field |
| --- | --- |
| OpenAI Chat, MiMo, MiniMax, Qwen | `max_completion_tokens` |
| OpenAI Responses | `max_output_tokens` |
| compat, DeepSeek, GLM, Grok, Ollama, OpenRouter | `max_tokens` |
| Anthropic | `max_tokens` (required by the API) |
| Gemini | `generationConfig.maxOutputTokens` |
| Bedrock | `inferenceConfig.maxTokens` |

## Thinking

`Thinking == nil` sends no thinking fields. `Effort` and `BudgetTokens` are sent as given. A combination the wire format cannot express returns a validation error before the request is sent.

| Provider | Enabled | `ThinkingDisabled` | `Effort` | `BudgetTokens` | `IncludeOutput` |
| --- | --- | --- | --- | --- | --- |
| OpenAI Chat, compat, Ollama | — | `reasoning_effort: "none"` | `reasoning_effort` | error | ignored |
| OpenAI Responses | — | `reasoning.effort: "none"` | `reasoning.effort` | error | `reasoning.summary: "auto"` |
| Anthropic | `thinking.type: "adaptive"` | `thinking.type: "disabled"` | `output_config.effort` | `thinking.type: "enabled"` + `budget_tokens` | `thinking.display: "summarized"` |
| Bedrock | as Anthropic, inside `additionalModelRequestFields` | same | same | same | same |
| Gemini | — | `thinkingConfig.thinkingBudget: 0` | `thinkingConfig.thinkingLevel` | `thinkingConfig.thinkingBudget` | `thinkingConfig.includeThoughts` |
| DeepSeek, GLM | `thinking.type: "enabled"` | `thinking.type: "disabled"` | `reasoning_effort` | error | ignored |
| Grok | — | error (cannot be disabled) | `reasoning_effort` | error | ignored |
| MiMo | `thinking.type: "enabled"` | `thinking.type: "disabled"` | error | error | ignored |
| MiniMax | `thinking.type: "adaptive"` | `thinking.type: "disabled"` | error | error | ignored |
| OpenRouter | `reasoning.enabled: true` | `reasoning.effort: "none"` | `reasoning.effort` | `reasoning.max_tokens` (not with `Effort`) | ignored |
| Qwen | `enable_thinking: true` | `enable_thinking: false` | error | `thinking_budget` | ignored |

Bedrock sends Claude's thinking format. For other model families, set their fields through the `additionalModelRequestFields` option instead of `Thinking`.

## Reasoning Output

Reasoning is returned as `ReasoningBlock`. Adapters read these fields, in priority order:

| Provider | Source |
| --- | --- |
| OpenAI Responses | `reasoning` items (summary, or `reasoning_text` content); the item is kept in `Extra` for replay |
| Anthropic, Bedrock | thinking blocks with signature; redacted thinking in `Redacted` |
| Gemini | `thought` parts; thought signatures on thought and text parts are kept in `Signature` |
| compat | `reasoning_content`, `reasoning` |
| DeepSeek, GLM, Grok, MiMo, Qwen | `reasoning_content` |
| MiniMax | `reasoning_details`, `reasoning_content` (requests set `reasoning_split: true`) |
| Ollama | `reasoning`, `reasoning_content`, `thinking` |
| OpenRouter | `reasoning_details`, `reasoning`, `reasoning_content`; `reasoning_details` is kept in `Extra` for replay |

When a `ReasoningBlock` is sent back in history, Chat Completions adapters write it to the first field above. OpenRouter replays `reasoning_details` from `Extra`; streamed fragments are merged into the entries a non-streaming response returns. OpenAI Responses replays only its own reasoning items from `Extra` and drops reasoning from other providers, since an input reasoning item needs its API-assigned id.

Gemini requires the thought signature of each function call in the current turn. History from another provider has none; set `ToolUseBlock.Signature` to `gemini.SkipThoughtSignatureValidator` to replay such calls.

## Cache Breakpoints

`CacheControl` on a block marks a breakpoint. It is a hint: adapters without a slot drop it.

| Provider | Encoding |
| --- | --- |
| OpenAI Chat and Responses | content part `prompt_cache_breakpoint: {"mode": "explicit"}`; a `TTL` is an error |
| Anthropic | `cache_control: {"type": "ephemeral", "ttl": TTL}` |
| OpenRouter | `cache_control: {"type": "ephemeral", "ttl": TTL}` on the content part |
| Bedrock | a `cachePoint: {"type": "default", "ttl": TTL}` block after the marked block |
| others | dropped; Gemini uses the `cachedContent` option |

## Provider Options

`ProviderOptions` carry native wire fields: each key is a top-level field of the vendor's request body. Keys are checked against the adapter's list and rejected when unknown, except in `compat`, which passes every key through. When a key names a field the adapter also generates, an object is merged into it and an array is appended to it; any other collision is an error. Generated JSON outside the merged objects is sent byte for byte, so schema property order is kept. The constants in each provider package name the accepted keys.
