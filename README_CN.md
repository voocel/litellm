# LiteLLM（Go）— 多平台 LLM API 客户端

[English](README.md) | 中文

LiteLLM 是一个小巧的 Go 客户端，用统一 API 访问多个 LLM 平台。

## 快速上手

### 安装

```bash
go get github.com/voocel/litellm
```

### 1) 准备 API Key

```bash
export OPENAI_API_KEY="your-key"
```

### 2) 创建 Client（显式 Provider）

```go
package main

import (
	"context"
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

	_, _ = client.Chat(context.Background(), &litellm.Request{
		Model: "gpt-4o-mini",
		Messages: []litellm.Message{
			{Role: "user", Content: "用一句话解释 AI。"},
		},
	})
}
```

> 说明
> - `providers` 包是内部实现细节，使用者只需要引入 `github.com/voocel/litellm`。
> - LiteLLM 不自动发现 Provider，也不自动路由模型，必须显式配置。

## 核心 API

- `New(provider, opts...)` 使用显式 Provider 创建客户端。
- `NewWithProvider(name, config, opts...)` 通过名称与配置创建客户端。
- `Request` 是跨平台统一的：必填 `Model` 与 `Messages`，可选 `MaxTokens`、`Temperature`、`TopP`、`Stop` 等控制项。
- `Chat(ctx, req)` 返回统一的 `Response`。
- `Stream(ctx, req)` 返回 `StreamReader`（不支持多 goroutine 并发读），务必 `defer stream.Close()`。

### 流式（最小示例）

```go
stream, err := client.Stream(ctx, &litellm.Request{
	Model: "gpt-4o-mini",
	Messages: []litellm.Message{
		{Role: "user", Content: "讲个笑话"},
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

## 高级能力（可选）

下面能力都可跨平台使用，更完整的可运行示例在 `examples/` 目录。

### 结构化输出

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
	Messages: []litellm.Message{{Role: "user", Content: "生成一个人。"}},
	ResponseFormat: litellm.NewResponseFormatJSONSchema("person", "", schema, true),
})
_ = resp
```

### 工具调用（Function Calling）

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
	Messages: []litellm.Message{{Role: "user", Content: "东京天气？"}},
	Tools: tools,
	ToolChoice: "auto",
})
_ = resp
```

### OpenAI Responses API

```go
resp, err := client.Responses(ctx, &litellm.OpenAIResponsesRequest{
	Model: "o3-mini",
	Messages: []litellm.Message{{Role: "user", Content: "逐步算 15*8"}},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	MaxOutputTokens:  litellm.IntPtr(800),
})
_ = resp
```

### 重试与超时

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

### 平台特定参数

`Request.Extra` 会按 Provider 进行校验，不支持的 Provider 会直接报错。

支持的键：
- Gemini：`tool_name`（string），用于 tool response 命名
- GLM：`enable_thinking`（bool）或 `thinking`（包含 `type` 的对象）

## 自定义 Provider

实现 `litellm.Provider` 并注册即可扩展平台：

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
	return nil, fmt.Errorf("未实现流式")
}

func init() {
	litellm.RegisterProvider("myprovider", func(cfg litellm.ProviderConfig) litellm.Provider {
		return &MyProvider{name: "myprovider", config: cfg}
	})
}
```

## 内置 Provider

已内置：OpenAI、Anthropic、Google Gemini、DeepSeek、Qwen（DashScope）、GLM、AWS Bedrock、OpenRouter。

LiteLLM 不会改写模型 ID，请使用官方模型 ID。

## 配置

显式配置 Provider：

```go
client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
	APIKey:  os.Getenv("OPENAI_API_KEY"),
	BaseURL: os.Getenv("OPENAI_BASE_URL"), // 可选
})
_ = client
```

## 许可证

Apache License
