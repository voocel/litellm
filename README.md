# LiteLLM Go

[中文](README_CN.md) | English

LiteLLM is a small, explicit Go SDK for calling LLM providers through one typed core model. The root package owns the provider-agnostic API; concrete providers live in `provider/<name>` subpackages.

## Install

```bash
go get github.com/voocel/litellm
```

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
	client, err := openai.NewClient(openai.Config{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "gpt-5.6",
		Messages: []litellm.Message{
			litellm.System("You are concise."),
			litellm.UserText("Explain Go interfaces in one sentence."),
		},
		MaxTokens: litellm.IntPtr(120),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
```

`openai.NewClient(cfg, opts...)` builds the provider first, then returns a ready `*litellm.Client`; every provider package exposes it. The explicit two-step form — `provider, _ := openai.New(cfg)` then `litellm.New(provider, opts...)` — is equivalent; prefer it when you want to share one provider across multiple clients. Both forms accept the same `ClientOption`s.

## Core Model

Messages and responses use ordered `Block` values:

- `TextBlock`
- `ImageBlock`
- `ReasoningBlock`
- `ToolUseBlock`
- `ToolResultBlock`
- `ToolReferenceBlock`

`Response.Blocks` is the canonical response content. `Text()`, `Reasoning()`, and `ToolCalls()` are convenience views.

```go
msgs := []litellm.Message{
	litellm.User(litellm.Text("What is in this image?"), litellm.ImageURL("https://example.com/cat.png")),
}

resp, err := client.Chat(ctx, litellm.Request{Model: "gpt-5.6", Messages: msgs})
_ = resp
_ = err
```

For multi-turn tool workflows, append the previous response blocks directly:

```go
args, err := litellm.JSONRaw(map[string]any{"ok": true})
if err != nil {
	log.Fatal(err)
}

msgs = append(msgs,
	litellm.Assistant(resp.Blocks...),
	litellm.ToolResultText("call_1", string(args)),
)
```

`JSONRaw` returns marshal errors instead of silently producing invalid tool arguments. Use `MustJSONRaw` only for static test data or package-level examples where panic is acceptable.

The Client validates the shared model's structure; Providers enforce their own protocol constraints. Applications explicitly check whether tool history is complete. Repair is a separate operation; `WithMessageRepair` has been removed:

```go
if err := litellm.ValidateHistory(msgs); err != nil {
    log.Fatal(err)
}

// Call only when the application chooses repair; msgs remains unchanged.
repaired, warnings := litellm.RepairMessages(msgs, litellm.RepairAll)
_ = repaired
_ = warnings
```

The application handles repair warnings directly. Provider normalization warnings still reach `Response.Warnings`, `WarningEvent`, and `CallObserver.OnEvent`. Synthetic tool results mark interrupted execution; they do not claim that a tool ran.

Raw provider response bodies are not retained by default. Enable them explicitly when debugging:

```go
client, err := openai.NewClient(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")}, litellm.WithCaptureRawResponse(true))
```

## Streaming

Streams emit typed `Event` values.
Providers with explicit content boundaries emit `ContentStart` / `ContentEnd`. The start carries initial content; the end may carry a complete final snapshot with metadata, not another delta. Snapshot text must match streamed text; mismatches return an error and the partial response. `Collect` handles both; `StreamText` / `StreamWith` deliver initial text and subsequent deltas.
`Stream` is intended for single-goroutine consumption; do not call `Next` concurrently.
Use `WithStreamIdleTimeout` when you want an explicit per-event idle timeout; it is off by default.
`WithStreamIdleTimeout` only covers generic `Client.Stream`; OpenAI Responses native streaming uses `openai.Config.StreamIdleTimeout`.
For example:

```go
client, err := openai.NewClient(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")}, litellm.WithStreamIdleTimeout(120*time.Second))
```

```go
stream, err := client.Stream(ctx, litellm.Request{
	Model:    "gpt-5.6",
	Messages: []litellm.Message{litellm.UserText("Tell me a short joke.")},
})
if err != nil {
	log.Fatal(err)
}
defer stream.Close()

for {
	event, err := stream.Next()
	if err != nil {
		log.Fatal(err)
	}
	switch e := event.(type) {
	case litellm.ContentStart:
		switch block := e.Block.(type) {
		case litellm.TextBlock:
			fmt.Print(block.Text)
		case litellm.ReasoningBlock:
			fmt.Print(block.Text)
		}
	case litellm.ContentDelta:
		fmt.Print(e.Text)
	case litellm.ReasoningDelta:
		fmt.Print(e.Text)
	case litellm.ProviderEvent:
		// Provider-native lifecycle/hosted-tool event.
	case litellm.DoneEvent:
		return
	}
}
```

To aggregate a stream (on failure, both a partial response and an error are returned; always check the error):

```go
resp, err := litellm.Collect(stream)
```

## Retry

Retries are off by default. `LiteLLMError.Temporary` / `IsTemporaryError` describe a potentially transient failure, not a guarantee that replay is safe. Opting into retries accepts possible duplicate requests and charges. The transport retries selected HTTP statuses, never network failures or interrupted response streams. Enable retries per provider:

```go
import "github.com/voocel/litellm/retry"

provider, err := openai.New(openai.Config{
	APIKey: os.Getenv("OPENAI_API_KEY"),
	Retry:  retry.DefaultPolicy(),
})
```

Bedrock retries re-sign each attempt internally, so users do not need to compose SigV4 transports by hand.

If you need a proxy, tracing, or a custom base transport, pass `Transport` together with `Retry`. A custom `HTTPClient` is an advanced escape hatch and cannot be combined with `Retry`; configure retry inside that client yourself.

Choose the smallest configuration that matches your use case:

| Use case | Config |
| --- | --- |
| Normal retries | `Retry: retry.DefaultPolicy()` |
| Retries plus proxy/tracing/custom base transport | `Retry` + `Transport` |
| Fully custom request execution | `HTTPClient`, without `Retry`/`Transport` |

`APIKeyFunc` is resolved once when a request is created; retry attempts reuse that request. If you use extremely short-lived Bearer tokens, inject auth in a lower-level custom `Transport` or `HTTPClient`. Normal API keys and the default retry window do not need special handling.

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

Thinking is explicit. If `Thinking` is nil, the SDK sends no thinking control fields.

```go
resp, err := client.Chat(ctx, litellm.Request{
	Model:    "claude-sonnet-5",
	Messages: []litellm.Message{litellm.UserText("Explain the tradeoffs.")},
	MaxTokens: litellm.IntPtr(2048),
	Thinking: &litellm.Thinking{
		Mode:  litellm.ThinkingEnabled,
		Effort: "low",
	},
})
```

Stable provider constraints are validated locally. Model-specific effort and disable limits are left to the provider API, so new models in the same API generation work without SDK updates.
Portable effort values are `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`, but support is model-specific.
Use `client.Capabilities(model)` or `litellm.GetCapabilities(provider, model)` for the stable UI/preflight baseline. Model-specific values outside that baseline can still be sent and are validated by the provider API.

## OpenAI Responses

Set `openai.Config.API = openai.APIResponses` to route generic `Client.Chat` and `Client.Stream` calls through the Responses API while keeping the shared `litellm.Request` and return types. The default is the Chat Completions API.

```go
client, err := openai.NewClient(openai.Config{
	APIKey: os.Getenv("OPENAI_API_KEY"),
	API:    openai.APIResponses,
})
```

For native fields such as hosted tools, conversation IDs, and `previous_response_id`, use `Responses` and `ResponsesStream` on `provider/openai.Provider`:

```go
oai, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
if err != nil {
	log.Fatal(err)
}

resp, err := oai.Responses(ctx, &openai.ResponsesRequest{
	Model: "gpt-5.6",
	Messages: []litellm.Message{
		litellm.UserText("Solve 15*8 step by step."),
	},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	ReasoningMode:    "pro",
	ReasoningContext: "all_turns",
	MaxOutputTokens:  litellm.IntPtr(800),
	OpenAITools: []openai.ResponsesTool{
		{"type": "web_search_preview"},
	},
})
```

Streaming Responses uses the same typed event model:

```go
oai, err := openai.New(openai.Config{
	APIKey:            os.Getenv("OPENAI_API_KEY"),
	StreamIdleTimeout: 120 * time.Second,
})

stream, err := oai.ResponsesStream(ctx, &openai.ResponsesRequest{
	Model:    "gpt-5.6",
	Messages: []litellm.Message{litellm.UserText("Search and summarize.")},
})
```

## Providers

Provider configs are provider-specific. Authentication is not forced into a single API-key shape.

```go
import (
	"github.com/voocel/litellm/provider/anthropic"
	"github.com/voocel/litellm/provider/bedrock"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/gemini"
	"github.com/voocel/litellm/provider/glm"
	"github.com/voocel/litellm/provider/grok"
	"github.com/voocel/litellm/provider/minimax"
	"github.com/voocel/litellm/provider/ollama"
	"github.com/voocel/litellm/provider/openrouter"
	"github.com/voocel/litellm/provider/qwen"
)
```

Examples:

```go
anthropic.New(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
gemini.New(gemini.Config{APIKey: os.Getenv("GEMINI_API_KEY")})
deepseek.New(deepseek.Config{APIKey: os.Getenv("DEEPSEEK_API_KEY")})
ollama.New(ollama.Config{})

bedrock.New(bedrock.Config{
	Region: "us-east-1",
	Credentials: bedrock.StaticCredentials(
		os.Getenv("AWS_ACCESS_KEY_ID"),
		os.Getenv("AWS_SECRET_ACCESS_KEY"),
		os.Getenv("AWS_SESSION_TOKEN"),
	),
})
```

Supported provider packages currently include OpenAI, Anthropic, Gemini, Bedrock, DeepSeek, Qwen, GLM, OpenRouter, MiniMax, Grok, MiMo, and Ollama.
See [Provider Capabilities](provider-capabilities.md) for thinking, reasoning, usage, and cache support across providers.

## Model Listing

```go
models, err := client.ListModels(ctx)
```

Only providers that implement `ModelLister` support this. Returned fields are best-effort.

## Provider Options

`Request.ProviderOptions` is `map[string]json.RawMessage`: it carries JSON data only. `NewProviderOptions` and `Set` encode values immediately and return encoding errors. The Client copies JSON bytes for execution and each Observer. Providers decode and validate their supported keys at their own boundary; unknown keys error by default.

```go
options, err := litellm.NewProviderOptions(map[string]any{
    openai.ProviderOptionPromptCacheOptions: openai.PromptCacheOptions{Mode: "implicit", TTL: "30m"},
})
if err != nil {
    log.Fatal(err)
}
resp, err := client.Chat(ctx, litellm.Request{
    Model: "gpt-5.6",
    Messages: []litellm.Message{litellm.UserText("Hello")},
    ProviderOptions: options,
})
```

`ToolChoice` no longer accepts strings or protocol objects. Use `&litellm.ToolChoice{Mode: litellm.ToolChoiceAuto}` (also `None` / `Required`), or `&litellm.ToolChoice{Name: "lookup"}` to select a named tool. `nil` leaves the provider default in place. Do not mutate requests concurrently with an invocation.

## Observers And OTel

An `Observer` starts a separate `CallObserver` for every Chat/Stream invocation, including local validation failures. `Start` receives an isolated snapshot of the caller's request before defaults and validation. Its returned context reaches later observers, the Provider and HTTP requests. Observer factories may run concurrently; each call owns its state.

`OnEvent` observes validated stream events and warnings (including Chat warnings). `End` runs once, in reverse observer registration order, with status `completed`, `failed`, `canceled` or `closed`, duration, error and the final/partial response. Opening a stream does not end the call. Consume streams to termination or Close them; cancellation without Next/Close does not run callbacks in the background.

Inputs, events and results are isolated copies. Callbacks run synchronously; the core does not recover panics. Application consumer callback errors belong to the consumer, not the model execution; closing that unfinished stream reports `closed`. A cleanup error after completion is returned by Close without revising the completed result. Deadlines and idle timeouts report `failed`; explicit context cancellation reports `canceled`.

```go
observer := litellm.ObserverFunc(func(ctx context.Context, info litellm.CallInfo) (context.Context, litellm.CallObserver) {
    return ctx, litellm.CallObserverFuncs{
        EndFunc: func(result litellm.CallResult) {
            fmt.Printf("%s/%s: %s (%s), err=%v\n",
                info.Provider, info.Model, result.Status, result.Duration, result.Err)
        },
    }
})
client, err := litellm.New(provider, litellm.WithObservers(observer))
```

The optional `github.com/voocel/litellm/otel` module creates a span per call and propagates its context to the transport. It uses the final/partial result without a global call registry, lock or duplicate stream collector. Content capture is off by default. Enable it explicitly to record messages, which may contain user data and tool arguments:

```go
import litellmotel "github.com/voocel/litellm/otel"

observer := litellmotel.New(tracer, litellmotel.WithCaptureContent(true))
client, err := litellm.New(provider, litellm.WithObservers(observer))
```

Migration: `Hook`, `HookFuncs`, `CallMeta` and `WithHook(s)` have been removed. Use `ObserverFunc`, `CallObserverFuncs`, `CallInfo`/`CallResult` and `WithObservers`; there is no compatibility shim. During development `otel/go.mod` replaces the core dependency with `..`. Before publishing, release the new core API, update OTel's required core version, and remove the local replacement.

## Usage

Token counts are `*int`: `nil` means unknown; `litellm.IntPtr(0)` means a known zero. Input includes cache reads and writes; output includes reasoning. Detail counts are subsets, not extra tokens to add again. Anthropic / Bedrock adapters add separate cache counts to input, and Gemini adds thoughts to output. Unreported details remain `nil`; OTel omits these attributes instead of recording zero.

```go
if resp.Usage.InputTokens != nil {
    fmt.Println(*resp.Usage.InputTokens)
}
```

Pricing requires known input and output counts. A distinct cache rate requires its corresponding cache count; missing data returns an error. Cache reads and writes are billed once, and negative counts or cache counts exceeding input are rejected. Unconfigured cache rates use the ordinary input rate.

## Pricing

Pricing is explicit. Cost calculation never loads remote pricing implicitly.

```go
import "github.com/voocel/litellm/pricing"

reg := pricing.NewRegistry()
err := reg.LoadFromURL(ctx, pricing.DefaultURL)
cost, err := reg.Calculate(resp.Model, resp.Usage)

err = reg.Set("my-model", pricing.ModelPricing{
	InputCostPerToken:  0.000001,
	OutputCostPerToken: 0.000002,
})
```

## Custom Providers

Implement the small provider interface:

```go
type Provider interface {
	Name() string
	Chat(context.Context, *litellm.Request) (*litellm.Response, error)
	Stream(context.Context, *litellm.Request) (litellm.Stream, error)
}
```

## License

Apache License
