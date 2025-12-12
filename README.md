# LiteLLM (Go) — Multi‑Provider LLM Client

[中文](README_CN.md) | English

LiteLLM is a small, typed Go client that lets you call multiple LLM providers through one API.

## Get Started

### Install

```bash
go get github.com/voocel/litellm
```

### 1) Set one API key

```bash
export OPENAI_API_KEY="your-key"
```

### 2) Call a model in one line

```go
package main

import (
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	resp, err := litellm.Quick("gpt-4o-mini", "Hello, LiteLLM!")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

### 3) Create a client (recommended for apps)

```go
package main

import (
	"context"
	"log"
	"os"

	"github.com/voocel/litellm"
)

func main() {
	client, err := litellm.New(
		litellm.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
		litellm.WithDefaults(1024, 0.7),
	)
	if err != nil {
		log.Fatal(err)
	}

	_, _ = client.Chat(context.Background(), &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "Explain AI in one sentence."},
		},
	})
}
```

> Notes
> - The `providers` subpackage is an internal implementation detail. End users should only import `github.com/voocel/litellm`.
> - Model strings are passed to the upstream API unchanged. Auto‑resolution only selects a provider, so prefer official model IDs.

## Core API

- `New(opts...)` builds a client. If you call `New()` with no options, it auto‑discovers providers from environment variables.
- `Request` is provider‑agnostic: set `Model` and `Messages`, then optional controls like `MaxTokens`, `Temperature`, `TopP`, `Stop`, etc.
- `Chat(ctx, req)` returns a unified `Response`.
- `Stream(ctx, req)` returns a `StreamReader` (not goroutine‑safe). Always `defer stream.Close()`.

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

for {
	chunk, err := stream.Next()
	if err != nil || chunk.Done {
		break
	}
	fmt.Print(chunk.Content)
}
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

### Reasoning models / Responses API (OpenAI)

```go
resp, err := client.Chat(ctx, &litellm.Request{
	Model: "o3-mini",
	Messages: []litellm.Message{{Role: "user", Content: "Solve 15*8 step by step."}},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	UseResponsesAPI:  true,
})
_ = resp
```

### Retries & timeouts

```go
client, _ := litellm.New(
	litellm.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
	litellm.WithRetries(3, 1*time.Second),
	litellm.WithTimeout(60*time.Second),
)
_ = client
```

### Provider‑specific knobs

Use `Request.Extra` for vendor‑specific parameters (e.g., Qwen/GLM thinking). See `examples/qwen`, `examples/glm`, and `examples/bedrock`.

## Custom Providers

Implement `litellm.Provider` and register it:

```go
type MyProvider struct {
	name   string
	config litellm.ProviderConfig
}

func (p *MyProvider) Name() string                     { return p.name }
func (p *MyProvider) Validate() error                 { return nil }
func (p *MyProvider) SupportsModel(model string) bool { return true }
func (p *MyProvider) Models() []litellm.ModelInfo {
	return []litellm.ModelInfo{
		{
			ID:              "my-model",
			Provider:        "myprovider",
			Name:            "My Model",
			MaxOutputTokens: 4096,
			Capabilities:    []litellm.ModelCapability{litellm.CapabilityChat},
		},
	}
}

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

LiteLLM does not rewrite model IDs; it only selects a provider. Always use official model IDs.

## Configuration

Environment variables for auto‑discovery:

```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export QWEN_API_KEY="sk-..."
export GLM_API_KEY="your-glm-key"
export OPENROUTER_API_KEY="sk-or-v1-..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

## License

Apache License
