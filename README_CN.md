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

### 2) 最小可运行示例

#### 文本对话

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
		Messages: []litellm.Message{litellm.UserMessage("用一句话解释 AI。")},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

#### 工具调用

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
		Messages:   []litellm.Message{litellm.UserMessage("东京天气怎么样？")},
		Tools:      tools,
		ToolChoice: "auto",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

#### 流式（收集）

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
		Messages: []litellm.Message{litellm.UserMessage("讲个笑话")},
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

如果你需要实时打印并最终得到聚合结果：

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

> 说明
> - `providers` 包是内部实现细节，使用者只需要引入 `github.com/voocel/litellm`。
> - LiteLLM 不自动发现 Provider，也不自动路由模型，必须显式配置。

## 核心 API

- `New(provider, opts...)` 使用显式 Provider 创建客户端。
- `NewWithProvider(name, config, opts...)` 通过名称与配置创建客户端。
- `Request` 是跨平台统一的：必填 `Model` 与 `Messages`，可选 `MaxTokens`、`Temperature`、`TopP`、`Stop` 等控制项。
- `Chat(ctx, req)` 返回统一的 `Response`。
- `Stream(ctx, req)` 返回 `StreamReader`（不支持多 goroutine 并发读），务必 `defer stream.Close()`。
- `CollectStream(stream)` 将流式结果收集为统一的 `Response`。
- `CollectStreamWithHandler(stream, onChunk)` 收集时也会处理每个 chunk。
- `CollectStreamWithCallbacks(stream, callbacks)` 提供内容/思考/工具回调。
- `Request.Thinking` 控制思考输出（默认开启，显式禁用才关闭）。
- `ListModels(ctx)` 列出当前 Provider 可用模型（仅部分 Provider 支持，字段为 best-effort）。

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

resp, err := litellm.CollectStream(stream)
if err != nil {
	log.Fatal(err)
}
fmt.Print(resp.Content)
```

## 高级能力（可选）

下面能力都可跨平台使用，更完整的可运行示例在 `examples/` 目录。

### 模型列表（部分 Provider 支持）

> 说明
> - 目前已支持：OpenAI / Anthropic / Gemini / OpenRouter / DeepSeek / Bedrock
> - 返回字段因平台差异而不同，`ModelInfo` 为 best-effort（可能为空）
> - Gemini 返回的模型名会自动去掉 `models/` 前缀，便于直接传给 `Request.Model`
> - Bedrock 可通过 `ProviderConfig.Extra["control_plane_base_url"]` 指定控制平面域名（默认从 `BaseURL` 推导）

```go
models, err := client.ListModels(ctx)
if err != nil {
	log.Fatal(err)
}
for _, m := range models {
	fmt.Println(m.ID, m.Name)
}
```

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

### 思考输出（默认开启）

```go
resp, err := client.Chat(ctx, &litellm.Request{
	Model:    "claude-haiku-4-5-20251001",
	Messages: []litellm.Message{litellm.UserMessage("说一句笑话。")},
	Thinking: litellm.NewThinkingEnabled(1024),
})
_ = resp
```

如需关闭：

```go
req := &litellm.Request{
	Model:    "claude-haiku-4-5-20251001",
	Messages: []litellm.Message{litellm.UserMessage("说一句笑话。")},
	Thinking: litellm.NewThinkingDisabled(),
}
_ = req
```

### OpenAI Responses API

```go
resp, err := client.Responses(ctx, &litellm.OpenAIResponsesRequest{
	Model: "o3-mini",
	Messages: []litellm.Message{{Role: "user", Content: "逐步算 15*8"}},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	Thinking:         litellm.NewThinkingEnabled(0),
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

### 费用计算

根据 token 使用量计算请求费用。定价数据来自 [BerriAI/litellm](https://github.com/BerriAI/litellm)，首次使用时自动加载。

```go
resp, err := client.Chat(ctx, req)
if err != nil {
	log.Fatal(err)
}

// 计算费用（定价数据自动加载）
if cost, err := litellm.CalculateCostForResponse(resp); err == nil {
	fmt.Printf("费用: $%.6f (输入: $%.6f, 输出: $%.6f)\n",
		cost.TotalCost, cost.InputCost, cost.OutputCost)
}

// 或使用独立函数
cost, err := litellm.CalculateCost(resp.Model, resp.Usage)

// 为未收录的模型设置自定义定价
litellm.SetModelPricing("my-model", litellm.ModelPricing{
	InputCostPerToken:  0.000001,
	OutputCostPerToken: 0.000002,
})
```

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
