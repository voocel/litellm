# Provider Mapping

Each adapter maps the shared `litellm.Request` onto its vendor's wire format, including the documented JSON Schema prompt fallback below. It does not infer what a model supports and does not check vendor values; whatever is sent is judged by the vendor API, and its error is returned as is. This page records the mapping so you can predict the request an adapter sends.

`client.Capabilities()` reports the static protocol facts at runtime: whether `MaxTokens` is required, whether `Thinking.Effort` and `Thinking.Disabled` can be sent (the "error" cells of the thinking table below), and the accepted `ProviderOptions` keys.

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

| Provider | Enabled | `Disabled` | `Effort` | `BudgetTokens` | `IncludeOutput` |
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

Anthropic also accepts a `thinking` provider option, for shapes litellm does not map: it is sent as given when `Thinking` is nil, and merged into the generated object otherwise, such as `{"display": "omitted"}`.

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
- **Gemini** requires a signature on the first function call of each model turn. A turn whose first call has none, such as one from another provider, gets the documented placeholder `skip_thought_signature_validator`, sent as that literal string. Signed text and thought parts retain their boundaries and never merge with neighboring parts, including a stream's trailing empty signature part. See [Thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures).
- **DeepSeek** requires the full `reasoning_content` of all previous assistant turns whenever a request carries `tools`, including turns without tool calls. Append `litellm.Assistant(resp.Blocks...)` to history after both `Chat` and collected streams. Without tools, DeepSeek ignores replayed reasoning. See [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/).
- **MiniMax, OpenRouter** stream `reasoning_details` in fragments; they are merged into the entries a non-streaming response returns.
- Empty text is not sent, and a message left with nothing to send, such as one holding only reasoning the target cannot carry, is omitted.
- A `ProviderState` with an empty `Provider` or invalid JSON `Data` fails validation.

## Structured Output

Callers use `ResponseFormatJSONSchema` with the same `JSONSchema` across providers. The adapter chooses how to send it:

| Provider | JSON Schema mapping |
| --- | --- |
| OpenAI, Anthropic, Gemini, Bedrock, Grok, Qwen, Ollama, OpenRouter, compat | Native schema field; availability and supported schema features depend on the model and endpoint |
| DeepSeek, GLM, MiMo | Schema in a prompt + `response_format: {"type":"json_object"}` |
| MiniMax | Schema in a prompt; no `response_format` field |

The fallback appends the schema name, description and full document to the last user message, or adds a user message if none exists. It preserves the caller's messages and reasoning/tool history. Both `Chat` and `Stream` use this mapping; fallback requests return `litellm.schema_prompt_fallback` in `Response.Warnings` or as a `WarningEvent` (also retained by `Collect`).

Prompting is best effort, including with `Strict` set: JSON mode constrains JSON syntax, not schema adherence, and prompting alone guarantees neither. The SDK does not validate or retry generated output; callers needing schema guarantees must validate it. Native adapters do not automatically retry with prompting when a particular model rejects a schema request.

## Gemini Request Formats

The adapter uses `generateContent` and `streamGenerateContent`. JSON output maps to `generationConfig.responseFormat.text.mimeType: "APPLICATION_JSON"`; JSON Schema also sets `responseFormat.text.schema`, preserving the schema document. The MIME value follows the enum in the [REST reference](https://ai.google.dev/api/generate-content#TextResponseFormat), rather than the string used by legacy `responseMimeType`.

`generationConfig.candidateCount` in `ProviderOptions` must be 1 when supplied: `litellm.Response` represents one output. Tool results marked `IsError` always go under `functionResponse.response.error`, including JSON objects.

## DeepSeek Request Formats

The default example model is `deepseek-flash`. Per the [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/), system and assistant text blocks are concatenated without separators into a string; images in those roles are rejected. User messages support text and images. URL and inline images use `image_url`; set `ImageBlock.FileURI` to an uploaded `file-api-...` ID to send a `file` part with `file_id` (see [Vision](https://api-docs.deepseek.com/guides/vision/)). File upload itself is outside this provider's API.

The API supports `text` and `json_object`; the SDK maps `json_schema` to the prompt fallback described above. When using `json_object` directly, also instruct the model to produce JSON in a system or user message.

For strict tool calls, set `deepseek.Config.BaseURL` to `https://api.deepseek.com/beta` and set `Strict: new(true)` on every tool. The provider sends the selected strict flags and uses the configured endpoint; DeepSeek validates the tool schemas. See [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/).

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
| OpenAI Chat and Responses | content part `prompt_cache_breakpoint: {"mode": "explicit"}`; the `prompt_cache_retention` option sets the lifetime for the whole request |
| Anthropic | `cache_control: {"type": "ephemeral", "ttl": TTL}` |
| OpenRouter | `cache_control: {"type": "ephemeral"}` on the content part |
| Bedrock | a `cachePoint: {"type": "default", "ttl": TTL}` block after the marked block |
| others | dropped; Gemini uses the `cachedContent` option |

`TTL` is sent where the table shows it; elsewhere the breakpoint keeps the vendor's default lifetime. OpenRouter accepts a TTL too, but its usage does not tell hour-long writes, which cost more, from five-minute ones, so they could not be priced; it is not sent. Thinking blocks cannot carry a breakpoint ([Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)), so `ReasoningBlock` has none.

OpenAI Responses sends cached tool results as `input_text` content parts. A
breakpoint on `ToolResultBlock` marks the last part; breakpoints on individual
text blocks mark those parts. Unmarked results remain strings.

## Usage

`Usage.InputTokens` counts every prompt token, cache reads and writes
included. A count the vendor does not report is zero, and `catalog.Pricing.Cost`
prices the input outside the cache counts at the input rate, so a vendor that
reports no cache counts, such as MiniMax, is priced as uncached input.

| Provider | Cache reads | Cache writes |
| --- | --- | --- |
| Anthropic, Bedrock | reported | reported, hour-long writes split out from `cache_creation` and `cacheDetails` |
| OpenAI Chat and Responses | `cached_tokens` | `cache_write_tokens` when sent |
| Gemini | `cachedContentTokenCount` | not reported |
| DeepSeek, GLM, Qwen | reported | 0: caching carries no write charge; Qwen breakpoints are dropped, leaving implicit caching |
| other Chat Completions vendors, compat | `cached_tokens` or `prompt_cache_hit_tokens` when sent | `cache_write_tokens` when sent |

DeepSeek, Gemini, GLM and Qwen are checked against their recorded responses in
`provider/testdata/live`. `LITELLM_LIVE=1 LITELLM_RECORD=1 go test ./provider
-run TestLive` calls the vendors whose key is set and records them again.

## OpenAI Response Metadata

Chat Completions preserves message annotations and choice logprobs on the first
text block. `Logprobs` keeps the native `{content, refusal}` object; streams
concatenate token entries and deliver it with the annotations in `BlockEnd`.
Responses keeps its native per-part logprobs array. Streamed Responses text
blocks end at `response.output_item.done`, after the final message `phase` is
known, so replay state matches non-streaming replies.

`background: true` is supported only with Responses `Stream`. `Chat` returns a
completed reply and rejects background jobs before sending a request; the SDK
does not expose background job polling or stream resumption.

## Tool Results

A `ToolResultBlock` holds text, images and tool references on every provider:

| Provider | Images | Tool references |
| --- | --- | --- |
| Chat Completions adapters | text stays in the tool message; the images follow the turn's tool messages in one user message, each result's introduced by `The image of tool call <id>:` | text |
| OpenAI Responses | `function_call_output.output` becomes a content list of `input_text` and `input_image` parts; text-only output stays a string ([Function calling](https://developers.openai.com/api/docs/guides/function-calling)) | text |
| Anthropic | `tool_result` content | native `tool_reference` |
| Bedrock | `toolResult.content` image blocks, which AWS supports for Nova and Claude models ([ToolResultContentBlock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultContentBlock.html)) | text |
| Gemini | `functionResponse.parts` as `inlineData`, a multimodal function response of Gemini 3 models; images by URL or file are rejected ([Multimodal function responses](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)) | text |

A tool reference sent as text reads `Tool <name> is now available.`; the tool must be among the request's tools.

`ToolUseBlock.Arguments` is the text the model wrote. Anthropic, Bedrock and Gemini need a JSON object on the wire and return a validation error naming the call when the arguments are not one; the Chat Completions and Responses adapters send the text as it is.

## Stream Endings

A stream that ends before the vendor's terminal event (`message_stop`, a finish reason, `response.completed`, Bedrock `metadata`) returns `io.EOF` from the adapter, which the Client reports as a temporary network error: a fresh request may complete. A server fault reported inside a stream (`api_error`, `server_error`, `internalServerException`, `modelStreamErrorException`) is temporary, as its HTTP 5xx would be.

## Provider Options

`ProviderOptions` carry native wire fields: each key is a top-level field of the vendor's request body. Keys are checked against the adapter's list and rejected when unknown, except in `compat`, which passes every key through. When a key names a field the adapter also generates, an object is merged into it and an array is appended to it; any other collision is an error. Generated JSON outside the merged objects is sent byte for byte, so schema property order is kept. The constants in each provider package name the accepted keys.

Options whose output litellm drops are not offered: Anthropic server tools (`tools`), `mcp_servers`, `container` and `context_management`; OpenAI Responses hosted tools (`tools`), `max_tool_calls`, server-side compaction (`context_management`) and `previous_response_id`, which a `Response` carries no id for.
