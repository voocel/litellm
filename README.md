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

A provider is safe to share across clients. `litellm.New` accepts `ClientOption`s such as `WithObservers`, `WithStreamIdleTimeout` and `WithCaptureRawResponse`.

## Core Model

Messages and responses use ordered `Block` values: `TextBlock`, `ImageBlock`, `ReasoningBlock`, `ToolUseBlock`, `ToolResultBlock` and `ToolReferenceBlock`. `Response.Blocks` is the canonical content; `Text()`, `Reasoning()` and `ToolCalls()` are views.

```go
msgs := []litellm.Message{
	litellm.User(litellm.Text("What is in this image?"), litellm.ImageURL("https://example.com/cat.png")),
}
```

For multi-turn tool workflows, append the previous response blocks as they are; reasoning signatures and provider extras travel with them:

```go
msgs = append(msgs,
	litellm.Assistant(resp.Blocks...),
	litellm.ToolResultText("call_1", `{"ok":true}`),
)
```

The Client checks the shared model's structure and never rewrites history. Tool-call pairing and repair are conversation policy, owned by the layer that manages the session.

Raw provider response bodies are kept only with `litellm.WithCaptureRawResponse(true)`.

## Streaming

A stream is a sequence of blocks. `BlockStart` opens the block at `Index`, the position it takes in `Response.Blocks`; `TextDelta`, `ReasoningDelta` and `ToolUseDelta` grow it; `BlockEnd` closes it with the completed block, including late metadata such as signatures. Blocks may interleave, and all end before `DoneEvent`. `UsageEvent`, `WarningEvent` and `ProviderEvent` (native events without a typed equivalent) carry the rest.

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

`Client.Stream` aggregates as events are read, so `Handle` and `Collect` return the complete response even after some events were read with `Next`. A stream is consumed by one goroutine. `WithStreamIdleTimeout` sets an optional per-event idle timeout.

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
tool.Strict = litellm.StrictEnabled

resp, err := client.Chat(ctx, litellm.Request{
	Model:      "gpt-5.6",
	Messages:   []litellm.Message{litellm.UserText("Weather in Paris?")},
	Tools:      []litellm.Tool{tool},
	ToolChoice: &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto},
})
```

`ToolChoice` takes a `Mode` (`Auto`, `None`, `Required`) or a tool `Name`; nil leaves the vendor default.

## Structured Output

```go
format, err := litellm.NewResponseFormatJSONSchema("person", "", map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name": map[string]any{"type": "string"},
	},
	"required": []string{"name"},
}, litellm.StrictEnabled)
if err != nil {
	log.Fatal(err)
}

resp, err := client.Chat(ctx, litellm.Request{
	Model:          "gpt-5.6",
	Messages:       []litellm.Message{litellm.UserText("Generate a person.")},
	ResponseFormat: format,
})
```

## Thinking

`Thinking == nil` sends no thinking fields and keeps the vendor default. Otherwise the zero `Mode` enables thinking and `ThinkingDisabled` turns it off; `Effort` and `BudgetTokens` are sent as given, and `IncludeOutput` asks for reasoning text where the vendor makes it optional.

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

Mark a cache breakpoint on a block; the prompt prefix up to and including it may be cached. `TTL` is passed as is (`litellm.CacheTTL5m`, `litellm.CacheTTL1h`, or empty for the vendor default). Breakpoints are hints: providers without a slot drop them.

```go
litellm.User(litellm.TextBlock{Text: longDocument, Cache: &litellm.CacheControl{TTL: litellm.CacheTTL1h}})
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

`client.Capabilities()` reports what the adapter can express (`ok` is false for a custom provider that declares nothing): whether `Thinking`, `ThinkingDisabled`, `Effort` and `BudgetTokens` are sent, and the accepted option keys. It is static per provider; whether a model honors a request is still the vendor's call.

## Providers

| Package | API |
| --- | --- |
| `provider/openai` | OpenAI Chat Completions (default) or Responses |
| `provider/anthropic` | Anthropic Messages |
| `provider/gemini` | Gemini `generateContent` |
| `provider/bedrock` | Amazon Bedrock Converse (SigV4) |
| `provider/deepseek`, `glm`, `grok`, `mimo`, `minimax`, `ollama`, `openrouter`, `qwen` | each vendor's Chat Completions dialect |
| `provider/compat` | any other OpenAI-compatible endpoint (vLLM, LM Studio, gateways) |

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

`openai` speaks the official protocol only (`max_completion_tokens`, `prompt_cache_breakpoint`). `compat` sends `max_tokens` and passes every provider option through unchecked, since it cannot know the server's fields.

### OpenAI Responses

Set `openai.Config.API = openai.APIResponses` to route `Chat` and `Stream` through the Responses API with the same request and response types. Native Responses fields are provider options; using an option of the other API is an error.

```go
provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY"), API: openai.APIResponses})

options, err := litellm.NewProviderOptions(map[string]any{
	openai.ProviderOptionPreviousResponseID: "resp_123",
	openai.ProviderOptionTools:              []any{map[string]any{"type": "web_search"}},
})
```

Hosted tools in `tools` are appended to the generated function tools; `text` and `reasoning` objects merge into the generated ones.

## Model Listing

```go
models, err := client.ListModels(ctx)
```

Available when the provider implements `ModelLister`. Returned fields are best-effort.

## Errors

HTTP failures and errors inside a stream are classified the same way; check with the `Is*` helpers instead of matching messages:

```go
switch {
case litellm.IsContextOverflowError(err):
	// compact history and resend
case litellm.IsRateLimitError(err), litellm.IsOverloadedError(err):
	time.Sleep(litellm.RetryAfter(err)) // 0 when the provider sent no Retry-After
case litellm.IsContentFilterError(err):
	// do not retry
}
```

Messages render as `provider: code: message`. Context overflow and content filtering are detected from vendor codes and messages even when a proxy rewrites the status; they are never marked temporary.

## Retry

Providers never retry. `Error.Temporary` describes a possibly transient failure, not permission to replay: the vendor may already have processed and billed the request. To opt in, wrap the HTTP client you pass to the provider:

```go
import "github.com/voocel/litellm/retry"

provider, err := openai.New(openai.Config{
	APIKey:     os.Getenv("OPENAI_API_KEY"),
	HTTPClient: retry.NewHTTPClient(nil, retry.DefaultPolicy()),
})
```

The transport retries complete 429, 500, 502, 503, 504 and 529 responses, never network failures or interrupted streams. A request body that cannot be resent returns the original response. For Bedrock, the retried request carries its SigV4 signature, which stays valid for five minutes.

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

During development `otel/go.mod` replaces the core module with `..`; release the core first, then update OTel's requirement and drop the replacement.

## Usage And Pricing

Token counts are `*int`: nil is unknown, `new(0)` a known zero. `Input()`, `Output()`, `Total()`, `Reasoning()`, `CacheRead()` and `CacheWrite()` return `(count, known)`. Input includes cache reads and writes; output includes reasoning; detail counts are subsets.

Pricing is explicit and never loads remote data implicitly:

```go
import "github.com/voocel/litellm/pricing"

reg := pricing.NewRegistry()
err := reg.LoadFromURL(ctx, pricing.DefaultURL)
cost, err := reg.Cost(resp.Model, resp.Usage)
```

## Custom Providers

Implement the provider interface; `CapabilityProvider` and `ModelLister` are optional:

```go
type Provider interface {
	Name() string
	Chat(context.Context, *litellm.Request) (*litellm.Response, error)
	Stream(context.Context, *litellm.Request) (litellm.Stream, error)
}
```

## License

Apache License
