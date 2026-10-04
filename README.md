# LiteLLM Go

[中文](README_CN.md) | English

LiteLLM is a small, explicit Go SDK for calling LLM providers through one typed core model. The root package owns the provider-agnostic API; concrete providers live in `provider/<name>` subpackages.

The SDK maps structure only. It does not infer what a model supports, does not validate vendor values locally, and does not rewrite your input: the vendor API decides, and its error is returned as a typed `*litellm.Error`.

## Install

```bash
go get github.com/voocel/litellm
```

Requires Go 1.26 or newer.

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/openai"
)

func main() {
	provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
	if err != nil {
		log.Fatal(err)
	}
	client, err := litellm.New(provider)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "gpt-5.6",
		Messages: []litellm.Message{
			litellm.System("You are concise."),
			litellm.UserText("Explain Go interfaces in one sentence."),
		},
		MaxTokens: new(120),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
```

A provider is safe to share across clients. `litellm.New` accepts `ClientOption`s such as `WithObservers` and `WithCaptureRawResponse`.

## Core Model

Messages and responses use ordered `Block` values: `TextBlock`, `ImageBlock`, `ReasoningBlock`, `ToolUseBlock`, `ToolResultBlock` and `ToolReferenceBlock`. `Response.Blocks` is the canonical content; `Text()`, `Reasoning()` and `ToolCalls()` are views.

```go
msgs := []litellm.Message{
	litellm.User(litellm.Text("What is in this image?"), litellm.ImageURL("https://example.com/cat.png")),
}
```

For multi-turn tool workflows, append the previous response blocks as they are, and keep them whole when you store history:

```go
msgs = append(msgs,
	litellm.Assistant(resp.Blocks...),
	litellm.ToolResultText("call_1", `{"ok":true}`),
)
```

Data a vendor needs back, such as a reasoning signature or an item id, travels in a block's `State` and is sent only to the provider that produced it, so history can move between providers ([details](providers.md#replay-state)). The Client checks the structure of messages and never rewrites history. Tool-call pairing and repair are conversation policy, owned by the layer that manages the session.

Messages encode as JSON with each block tagged by its `type`, state included, so `json.Marshal` stores history and `json.Unmarshal` reads it back as it was.

Raw provider response bodies are kept only with `litellm.WithCaptureRawResponse(true)`.

## Streaming

A stream is a sequence of blocks. `BlockStart` opens the block at `Index`, the position it takes in `Response.Blocks`; `TextDelta`, `ReasoningDelta` and `ToolUseDelta` grow it; `BlockEnd` closes it with the completed block, including metadata that arrives late, such as `State`. Blocks may interleave, and all end before `DoneEvent`. `UsageEvent`, `WarningEvent` and `ProviderEvent` (native events without a typed equivalent) carry the rest.

`Handle` aggregates a stream and passes each event to a callback; `Collect` only aggregates. On failure both return the partial response with the error.

```go
stream, err := client.Stream(ctx, litellm.Request{
	Model:    "gpt-5.6",
	Messages: []litellm.Message{litellm.UserText("Tell me a short joke.")},
})
if err != nil {
	log.Fatal(err)
}
defer stream.Close()

resp, err := litellm.Handle(stream, func(event litellm.Event) error {
	switch e := event.(type) {
	case litellm.ReasoningDelta:
		fmt.Print(e.Text)
	case litellm.TextDelta:
		fmt.Print(e.Text)
	}
	return nil
})
```

`Client.Stream` aggregates as events are read, so `Handle` and `Collect` return the complete response even after some events were read with `Next`. A stream is consumed by one goroutine.

A deadline on `ctx` bounds the whole call, which for a model that thinks at length must be long. To also end a stream over a connection that hung, `litellm.WithStreamIdleTimeout(d)` fails it, with a temporary network error, once it waits `d` for data: for the response, retries and their backoff included, or then for the body. Any data counts, vendor pings and gateway heartbeats included, so set `d` above the longest silence of a healthy stream, as a vendor may be silent while the model reads a long prompt or thinks. With retries, the wait for the response spans every attempt and backoff.

## Tools

```go
tool, err := litellm.NewTool("get_weather", "Get weather for a city.", map[string]any{
	"type": "object",
	"properties": map[string]any{
		"city": map[string]any{"type": "string"},
	},
	"required": []string{"city"},
})
if err != nil {
	log.Fatal(err)
}
tool.Strict = new(true)

resp, err := client.Chat(ctx, litellm.Request{
	Model:      "gpt-5.6",
	Messages:   []litellm.Message{litellm.UserText("Weather in Paris?")},
	Tools:      []litellm.Tool{tool},
	ToolChoice: &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto},
})
```

`ToolChoice` takes a `Mode` (`Auto`, `None`, `Required`) or a tool `Name`; nil leaves the vendor default.

`ToolUseBlock.Arguments` is the text the model wrote. It is meant to be a JSON object but may not be one, as when the reply was cut off at the output limit; the Client then adds a `litellm.tool_arguments_invalid` warning, and providers whose wire format needs an object reject the call in history with a validation error naming it.

A tool result holds text, images and tool references on every provider. Where a tool result carries text only, as in Chat Completions, the images follow the turn's tool messages in a user message.

A `Tool` marked `Deferred` is offered once a `ToolReferenceBlock` in a tool result names it, as a tool search returns; `Request.OfferedTools` lists what a request offers. Anthropic receives every tool from the first request on, deferred ones with `defer_loading`, so the tools of a conversation never change and its prompt cache and thinking stay valid; at least one tool must not be deferred. Bedrock, which cannot defer a tool, receives them all; the other providers receive the offered tools.

## Structured Output

```go
format, err := litellm.NewResponseFormatJSONSchema("person", "", map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name": map[string]any{"type": "string"},
	},
	"required": []string{"name"},
})
if err != nil {
	log.Fatal(err)
}
format.JSONSchema.Strict = new(true)

resp, err := client.Chat(ctx, litellm.Request{
	Model:          "gpt-5.6",
	Messages:       []litellm.Message{litellm.UserText("Generate a person.")},
	ResponseFormat: format,
})
```

## Thinking

`Thinking == nil` sends no thinking fields and keeps the vendor default. Otherwise thinking is on, and `Disabled` turns it off; `Effort` and `BudgetTokens` are sent as given, and `IncludeOutput` asks for reasoning text where the vendor makes it optional.

```go
resp, err := client.Chat(ctx, litellm.Request{
	Model:     "claude-sonnet-5",
	Messages:  []litellm.Message{litellm.UserText("Explain the tradeoffs.")},
	MaxTokens: new(2048),
	Thinking:  &litellm.Thinking{Effort: "low"},
})
```

Which values a model accepts is the vendor's decision. [providers.md](providers.md) lists the exact wire mapping per provider.

## Prompt Caching

Mark a cache breakpoint on a block; the prompt prefix up to and including it may be cached. `TTL` sets how long, passed as is, such as `"1h"`; empty is the vendor's default, five minutes on Anthropic and Bedrock. Breakpoints are hints: providers without a slot drop them, and those that cannot send a TTL use the default ([providers.md](providers.md#cache-breakpoints)).

```go
litellm.User(litellm.TextBlock{Text: longDocument, Cache: &litellm.CacheControl{TTL: "1h"}})
```

## Provider Options

`Request.ProviderOptions` carries native wire fields: each key is a top-level field of the vendor's request body, and its value is JSON. Keys are checked against the provider's list and rejected when unknown. When a key names a field the adapter also generates, an object is merged into it and an array is appended to it; any other collision is an error.

```go
options, err := litellm.NewProviderOptions(map[string]any{
	openai.ProviderOptionPromptCacheKey: "session-42",
	openai.ProviderOptionServiceTier:    "flex",
})
if err != nil {
	log.Fatal(err)
}
resp, err := client.Chat(ctx, litellm.Request{
	Model:           "gpt-5.6",
	Messages:        []litellm.Message{litellm.UserText("Hello")},
	ProviderOptions: options,
})
```

## Providers

| Package | API |
| --- | --- |
| `provider/openai` | OpenAI Chat Completions (default) or Responses |
| `provider/anthropic` | Anthropic Messages |
| `provider/gemini` | Gemini `generateContent` |
| `provider/bedrock` | Amazon Bedrock Converse (SigV4) |
| `provider/deepseek`, `glm`, `grok`, `mimo`, `minimax`, `ollama`, `openrouter`, `qwen` | each vendor's Chat Completions dialect |
| `provider/compat` | any other OpenAI-compatible endpoint (vLLM, LM Studio, gateways) |
| `gateway` | a litellm [gateway](#gateway) |

```go
anthropic.New(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
gemini.New(gemini.Config{APIKey: os.Getenv("GEMINI_API_KEY")})
ollama.New(ollama.Config{})
compat.New(compat.Config{BaseURL: "http://localhost:8000/v1"})

bedrock.New(bedrock.Config{
	Region: "us-east-1",
	Credentials: bedrock.StaticCredentials(
		os.Getenv("AWS_ACCESS_KEY_ID"),
		os.Getenv("AWS_SECRET_ACCESS_KEY"),
		os.Getenv("AWS_SESSION_TOKEN"),
	),
})
```

Applications that choose a provider from configuration build it by name; `provider.Config` holds the shared settings:

```go
import "github.com/voocel/litellm/provider"

provider, err := provider.New("anthropic", provider.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
names := provider.Names() // "anthropic", "bedrock", "compat", ...
```

`openai` follows the official protocol only. `compat` is for any other OpenAI-compatible server and passes provider options through unchecked, since it cannot know the server's fields.

Each provider's `Config.Name`, or `provider.Config.Name`, renames it: responses, errors and replay state carry the name. Give each endpoint of one protocol its own, such as an Anthropic-compatible proxy, so a reasoning signature is replayed only where it was issued.

`client.Capabilities()` reports the protocol facts of the adapter: whether `MaxTokens` is required, whether it sends `Thinking.Effort` and `Thinking.Disabled`, and the accepted option keys. It is static per provider, and `ok` is false for a provider that declares nothing, such as a gateway. Which models think is model data, in the [catalog](#usage-and-model-catalog); whether a model honors a request is the vendor's call.

### OpenAI Responses

Set `openai.Config.API = openai.APIResponses` to route `Chat` and `Stream` through the Responses API with the same request and response types. Native Responses fields are provider options; using an option of the other API is an error. Hosted tools and server-side compaction are not offered, since litellm does not model their output, nor is `previous_response_id`: history is sent whole.

```go
provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY"), API: openai.APIResponses})

options, err := litellm.NewProviderOptions(map[string]any{
	openai.ProviderOptionStore:   false,
	openai.ProviderOptionInclude: []any{"reasoning.encrypted_content"},
})
```

## Gateway

Package `gateway` runs calls on a gateway that holds the vendor keys, such as for an agent in a sandbox that must never see one. The client is an ordinary provider, which `provider.New("gateway", …)` also builds; the gateway serves `gateway.Server`, whose `Route` picks the Client for each call and may rewrite the request:

```go
// gateway
srv := &gateway.Server{Route: func(r *http.Request, req *litellm.Request) (*litellm.Client, error) {
	if req.Model != "smart" {
		return nil, errors.New("unknown model")
	}
	req.Model = "claude-sonnet-4-5"
	return anthropicClient, nil
}}
http.Handle("/v1/llm", auth(srv))

// sandbox
p, _ := gateway.New(gateway.Config{BaseURL: "https://gw.example.com/v1/llm", APIKey: teamToken})
```

Calls keep everything on the way: the request whole but for the key, blocks with their state, and errors with their type, retry facts and upstream provider. The upstream rejecting the gateway's own vendor key is the exception: it reaches the caller as a provider error, "upstream key rejected", rather than an auth error, which would blame the caller's key. The Server neither authenticates nor meters: put the caller on the request context in your authentication, and meter with an Observer on the routed Clients, which see that context.

Replies stream, so middleware in front of the Server must keep the ResponseWriter flushable, by implementing `http.Flusher` or `Unwrap`; the Server refuses calls otherwise. While the upstream is silent, before it answers too, the Server writes a heartbeat line every 15 seconds, which keeps proxies from cutting the connection and which the client skips. A call refused before the first line, by `Route` or the upstream, is answered with the HTTP status of its error, the upstream's or else one for its type such as 429 for a rate limit, and a `Retry-After` when it suggests a wait; a failure after it is an error event in the stream. It refuses request bodies over `gateway.MaxRequestBytes` (64 MiB). The caller cannot see the vendor's capabilities, so `Route` sets `MaxTokens` where the vendor requires it.

## Errors

HTTP failures and errors inside a stream are classified the same way; switch on `ErrorTypeOf` instead of matching messages:

```go
switch litellm.ErrorTypeOf(err) {
case litellm.ErrorTypeContextOverflow:
	// compact history and resend
case litellm.ErrorTypeRateLimit, litellm.ErrorTypeOverloaded:
	time.Sleep(litellm.RetryAfter(err)) // 0 when the provider sent no Retry-After
case litellm.ErrorTypeContentFilter:
	// do not retry
case litellm.ErrorTypeCanceled:
	// the caller cancelled; errors.Is(err, context.Canceled) holds as well
}
```

`IsTemporaryError` reports a failure that a fresh request may avoid: rate limits, overload, network failures, a stream cut off before its end, and server faults, whether an HTTP 5xx or reported inside a stream. The HTTP status decides the type and the vendor's error code fills `Code`; messages render as `provider: code: message`. Context overflow, content filtering and exhausted quota are detected from vendor codes and messages even when a proxy rewrites the status; they are never marked temporary. `ErrorTypeTimeout` is the caller's own deadline, the context's or the HTTP client's; a server that timed out is a temporary provider error.

## Retry

Providers never retry. `Error.Temporary` describes a possibly transient failure, not permission to replay: the vendor may already have processed and billed the request. To opt in, wrap the HTTP client you pass to the provider:

```go
import "github.com/voocel/litellm/retry"

provider, err := openai.New(openai.Config{
	APIKey:     os.Getenv("OPENAI_API_KEY"),
	HTTPClient: retry.NewHTTPClient(nil, retry.DefaultPolicy()),
})
```

- Retried: complete 408, 429, 500, 502, 503, 504 and 529 responses, unless the body shows exhausted quota, an auth failure, content filtering or context overflow.
- Not retried, returning the original response or error: network failures, interrupted streams, requests whose body cannot be resent, and a `Retry-After` beyond `MaxRetryAfter` (60s by default).
- Bedrock: the retried request reuses its SigV4 signature, which stays valid for five minutes.

A caller retrying at another layer, such as a whole stream that broke off, can pace its retries with the same `Policy`: `Policy.Delay` returns the wait before the next attempt, or reports that the server's `Retry-After` ends retrying.

## Observers And OTel

An `Observer` starts a `CallObserver` for every Chat/Stream invocation, including local validation failures. `CallInfo.Request` is a snapshot of the caller's request, shared by all observers. `OnEvent` receives stream events (and Chat warnings) as they arrive; `End` runs once with the status (`completed`, `failed`, `canceled`, `closed`), duration, error and the final or partial response. Consume streams to termination or Close them.

```go
type logCall struct{ info litellm.CallInfo }

func (c logCall) OnEvent(litellm.Event) {}
func (c logCall) End(r litellm.CallResult) {
	log.Printf("%s/%s: %s in %s, err=%v", c.info.Provider, c.info.Request.Model, r.Status, r.Duration, r.Err)
}

observer := litellm.ObserverFunc(func(ctx context.Context, info litellm.CallInfo) (context.Context, litellm.CallObserver) {
	return ctx, logCall{info}
})
client, err := litellm.New(provider, litellm.WithObservers(observer))
```

The optional `github.com/voocel/litellm/otel` module creates a GenAI semantic-convention span per call and propagates its context to the transport. Content capture is off by default; enable it explicitly, since messages may contain user data:

```go
import litellmotel "github.com/voocel/litellm/otel"

observer := litellmotel.New(tracer, litellmotel.WithCaptureContent(true))
```

## Usage And Model Catalog

Token counts are plain ints; a count the vendor does not report is zero. Input includes cache reads and writes; output includes reasoning; detail counts are subsets, such as `CacheWrite1hTokens`, the writes cached for an hour. `Pricing.Cost` prices the input not read from or written to the cache at the input rate, so a vendor that reports no cache counts is priced as uncached input. Hour-long writes are priced at `CacheWrite1h`, and are an error without it. So is usage without input tokens: every call has some, so the vendor reported none, and its cost is unknown rather than zero. Vendors that charge more for long inputs price every token of such a call higher: `Pricing.Tiers`, loaded from the list's `*_above_<n>k_tokens` keys and `tiered_pricing` tables, hold those rates, and `Cost` uses the last tier the input tokens are above.

The catalog holds model facts (context window, output limit, reasoning support and prices) from LiteLLM's model list, and never loads remote data implicitly:

```go
import (
	"fmt"

	"github.com/voocel/litellm/catalog"
)

var models catalog.Catalog
if err := models.LoadFromURL(ctx, catalog.DefaultURL); err != nil {
	return err
}

model, ok := models.Get("claude-sonnet-4-5") // exact model-list key
if !ok {
	return fmt.Errorf("model not found in catalog")
}
if model.Pricing == nil {
	return fmt.Errorf("model pricing is unavailable")
}

cost, err := model.Pricing.Cost(resp.Usage)
if err != nil {
	return err
}
```

Names are the list's keys, whose vendor prefixes follow LiteLLM's provider names, such as `xai/`, `zai/` and `dashscope/`, rather than `provider.Names()`. The catalog does not translate them, since one model may be listed under several sites at different prices; keep the catalog name of each model you price. `provider.CatalogName(name, model)` gives the name for a built-in provider, such as `xai/grok-4` for `grok`.

`Get` matches the complete key exactly. If `vendor/model` is absent, it returns `ok == false` even when `model` exists.

`ok == false` means the model is absent; `Pricing == nil` means its price is unknown. A non-nil `Pricing` with zero rates means usage is free.

`Model.Reasoning` is `*bool`: nil means unknown, false means unsupported, and true means supported. Token limits of zero are unknown. `Set` and model-list loading validate names, limits and rates; a failed load leaves the catalog unchanged. Providers never consult the catalog to change requests.

## Custom Providers

Implement the provider interface; `CapabilityProvider` is optional:

```go
type Provider interface {
	Name() string
	Chat(context.Context, *litellm.Request) (*litellm.Response, error)
	Stream(context.Context, *litellm.Request) (litellm.Stream, error)
}
```

To test code that calls models, `litellmtest.New` returns a provider that streams scripted replies and records the requests it got.

## License

Apache License
