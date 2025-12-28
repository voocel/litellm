# LiteLLM (Go) — Multi‑Provider LLM Client

[中文](README_CN.md) | English

LiteLLM is a small, typed Go client that lets you call multiple LLM providers through one API.

## Get Started

### Install

```bash
go get github.com/voocel/litellm
```

### 1) Prepare an API key

```bash
export OPENAI_API_KEY="your-key"
```

### 2) Quick examples (minimal runnable)

#### Text (chat)

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), &litellm.Request{
		Model:    "gpt-4o-mini",
		Messages: []litellm.Message{litellm.NewUserMessage("Explain AI in one sentence.")},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

#### Tool calling

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	tools := []litellm.Tool{
		litellm.NewTool("get_weather", "Get weather for a city.", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		}),
	}

	resp, err := client.Chat(context.Background(), &litellm.Request{
		Model:      "gpt-4o-mini",
		Messages:   []litellm.Message{litellm.NewUserMessage("Weather in Tokyo?")},
		Tools:      tools,
		ToolChoice: "auto",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

#### Streaming (collect)

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
		APIKey: os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
		log.Fatal(err)
	}

	stream, err := client.Stream(context.Background(), &litellm.Request{
		Model:    "gpt-4o-mini",
		Messages: []litellm.Message{litellm.NewUserMessage("Tell me a joke.")},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()

resp, err := litellm.CollectStream(stream)
if err != nil {
	log.Fatal(err)
}
fmt.Println(resp.Content)
}
```

If you need real-time streaming and a final aggregated response:

```go
resp, err := litellm.CollectStreamWithHandler(stream, func(chunk *litellm.StreamChunk) {
	if chunk.Type == litellm.ChunkTypeContent && chunk.Content != "" {
		fmt.Print(chunk.Content)
	}
	if chunk.Reasoning != nil && chunk.Reasoning.Done {
		fmt.Print("\n[reasoning done]")
	}
})
if err != nil {
	log.Fatal(err)
}
fmt.Println("\n---")
fmt.Println(resp.Content)
```

> Notes
> - The `providers` package is an internal implementation detail. End users should only import `github.com/voocel/litellm`.
> - LiteLLM does not auto-discover providers or auto-route models. You must configure providers explicitly.

## Core API

- `New(provider, opts...)` builds a client with an explicit provider.
- `NewWithProvider(name, config, opts...)` builds a client from a provider name and config.
- `Request` is provider‑agnostic: set `Model` and `Messages`, then optional controls like `MaxTokens`, `Temperature`, `TopP`, `Stop`, etc.
- `Chat(ctx, req)` returns a unified `Response`.
- `Stream(ctx, req)` returns a `StreamReader` (not goroutine‑safe). Always `defer stream.Close()`.
- `CollectStream(stream)` collects a stream into a unified `Response`.
- `CollectStreamWithHandler(stream, onChunk)` collects and also handles each chunk.
- `CollectStreamWithCallbacks(stream, callbacks)` adds content/reasoning/tool callbacks.

### Streaming (minimal)

```go
stream, err := client.Stream(ctx, &litellm.Request{
	Model: "gpt-4o-mini",
	Messages: []litellm.Message{
		{Role: "user", Content: "Tell me a joke."},
	},
})
if err != nil {
	log.Fatal(err)
}
defer stream.Close()

resp, err := litellm.CollectStream(stream)
if err != nil {
	log.Fatal(err)
}
fmt.Print(resp.Content)
```

## Advanced Features (optional)

Each feature below works across providers. Longer runnable examples live in `examples/`.

### Structured outputs

```go
schema := map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name": map[string]any{"type": "string"},
		"age":  map[string]any{"type": "integer"},
	},
	"required": []string{"name", "age"},
}

resp, err := client.Chat(ctx, &litellm.Request{
	Model: "gpt-4o-mini",
	Messages: []litellm.Message{{Role: "user", Content: "Generate a person."}},
	ResponseFormat: litellm.NewResponseFormatJSONSchema("person", "", schema, true),
})
_ = resp
```

### Function calling

```go
tools := []litellm.Tool{
	{
		Type: "function",
		Function: litellm.FunctionDef{
			Name: "get_weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
		},
	},
}

resp, err := client.Chat(ctx, &litellm.Request{
	Model: "gpt-4o-mini",
	Messages: []litellm.Message{{Role: "user", Content: "Weather in Tokyo?"}},
	Tools: tools,
	ToolChoice: "auto",
})
_ = resp
```

### OpenAI Responses API

```go
resp, err := client.Responses(ctx, &litellm.OpenAIResponsesRequest{
	Model: "o3-mini",
	Messages: []litellm.Message{{Role: "user", Content: "Solve 15*8 step by step."}},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	MaxOutputTokens:  litellm.IntPtr(800),
})
_ = resp
```

### Retries & timeouts

```go
res := litellm.DefaultResilienceConfig()
res.MaxRetries = 3
res.InitialDelay = 1 * time.Second
res.RequestTimeout = 60 * time.Second

client, _ := litellm.NewWithProvider("openai", litellm.ProviderConfig{
	APIKey:     os.Getenv("OPENAI_API_KEY"),
	Resilience: res,
})
_ = client
```

### Provider‑specific knobs

`Request.Extra` is validated per provider. Unsupported providers will return an error.

Supported keys:
- Gemini: `tool_name` (string) for tool response naming
- GLM: `enable_thinking` (bool) or `thinking` (object with `type` string)

## Custom Providers

Implement `litellm.Provider` and register it:

```go
type MyProvider struct {
	name   string
	config litellm.ProviderConfig
}

func (p *MyProvider) Name() string                     { return p.name }
func (p *MyProvider) Validate() error                 { return nil }

func (p *MyProvider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	return &litellm.Response{Content: "hello", Model: req.Model, Provider: p.name}, nil
}
func (p *MyProvider) Stream(ctx context.Context, req *litellm.Request) (litellm.StreamReader, error) {
	return nil, fmt.Errorf("streaming not implemented")
}

func init() {
	litellm.RegisterProvider("myprovider", func(cfg litellm.ProviderConfig) litellm.Provider {
		return &MyProvider{name: "myprovider", config: cfg}
	})
}
```

## Supported Providers

Builtin providers: OpenAI, Anthropic, Google Gemini, DeepSeek, Qwen (DashScope), GLM, AWS Bedrock, OpenRouter.

LiteLLM does not rewrite model IDs. Always use official model IDs.

## Configuration

Configure providers explicitly:

```go
client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
	APIKey:  os.Getenv("OPENAI_API_KEY"),
	BaseURL: os.Getenv("OPENAI_BASE_URL"), // optional
})
_ = client
```

## License

Apache License
