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
| OpenAI Responses | `reasoning` items (summary, or `reasoning_text` content) |
| Anthropic, Bedrock | thinking and redacted thinking blocks |
| Gemini | `thought` parts |
| compat | `reasoning_content`, `reasoning` |
| DeepSeek, GLM, Grok, MiMo, Qwen | `reasoning_content` |
| MiniMax | `reasoning_details`, `reasoning_content` (requests set `reasoning_split: true`) |
| Ollama | `reasoning`, `reasoning_content`, `thinking` |
| OpenRouter | `reasoning_details`, `reasoning`, `reasoning_content` |

Encrypted or redacted reasoning has empty `Text` and a non-nil `State`.

## Replay State

Some output must go back to its vendor verbatim in the next request: a reasoning signature, encrypted reasoning, an item id. Adapters keep it in the block's `State`, a `ProviderState` naming the provider and requested model, with the vendor's native fields as `Data`. Store it with the history and do not interpret it.

The replay rule is the same for every adapter: portable content (text, reasoning text, tool calls) is mapped wherever the wire format can carry it, and `State` is sent only to the provider that produced it. A block the wire format cannot carry without its `State` is dropped, and where the vendor documents a placeholder for foreign content, the adapter supplies it. Blocks built by the caller have no `State` and count as foreign.

| Provider | `State.Data` | Reasoning from elsewhere |
| --- | --- | --- |
| Anthropic | thinking: `{"type":"thinking","signature":…}`, redacted: `{"type":"redacted_thinking","data":…}` | dropped: thinking needs a Claude signature |
| Bedrock | `{"signature":…}` or `{"redactedContent":…}`, `{}` when unsigned | dropped |
| Gemini | `{"thoughtSignature":…}` on text, thought and function call blocks | sent as an unsigned thought |
| OpenAI Responses | reasoning: the `reasoning` item; text: `{"id":…,"phase":…}`; tool calls: `{"id":…}` | dropped: an input reasoning item needs its API-assigned id |
| MiniMax, OpenRouter | reasoning: the merged `reasoning_details` array | `Text` goes to the text reasoning field |
| other Chat Completions | none | `Text` goes to the first reasoning field above |

- **OpenAI Responses** pairs a reasoning item with the ids of the items after it, and only the model that produced the reasoning accepts it. An assistant message holding reasoning from the requested model (`State.Model == Request.Model`) is replayed whole, reasoning and ids included; any other is sent as plain content without reasoning or ids. A message's `phase` is always kept.
- **Gemini** requires a signature on the first function call of each model turn. A turn whose first call has none, such as one from another provider, gets the documented placeholder `skip_thought_signature_validator`, sent as that literal string.
- **MiniMax, OpenRouter** stream `reasoning_details` in fragments; they are merged into the entries a non-streaming response returns.
- Empty text is not sent, and a message left with nothing to send, such as one holding only reasoning the target cannot carry, is omitted.
- A `ProviderState` with an empty `Provider` or invalid JSON `Data` fails validation.

## System Messages

System messages before the first other message go to the vendor's top-level system field (for OpenAI Responses, up to the first one with a cache breakpoint). Later ones stay in place where the wire format has a slot, so that changing an instruction mid-conversation keeps the cached prefix and earlier thinking valid:

| Provider | Leading | Later |
| --- | --- | --- |
| Chat Completions adapters | `system` message | `system` message in place |
| OpenAI Responses | `instructions` (a `developer` message when it has a cache breakpoint) | `developer` message in place |
| Anthropic | `system` | `role: "system"` message in place |
| Gemini, Bedrock | `systemInstruction` / `system` | hoisted to the same field |

Anthropic accepts mid-conversation system messages only on some models (not Sonnet 5, for example), and only right after a user turn and before an assistant turn or at the end; elsewhere the API returns an error. Put such instructions first, or in a user message, for other models and placements.

## Tool Call IDs

Anthropic requires tool call ids matching `[a-zA-Z0-9_-]+`, and Bedrock Converse accepts at most 64 characters. Both adapters send conforming ids unchanged and rewrite any other, such as Kimi's `functions.lookup:0`, deterministically: disallowed characters become `_` and a hash of the original id is appended, so distinct ids stay distinct. Tool calls and tool results are rewritten alike, so they stay paired.

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
