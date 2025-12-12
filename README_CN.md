# LiteLLM（Go）— 多平台 LLM API 客户端

[English](README.md) | 中文

LiteLLM 是一个小巧的 Go 客户端，用统一 API 访问多个 LLM 平台。

## 快速上手

### 安装

```bash
go get github.com/voocel/litellm
```

### 1) 设置一个 API Key

```bash
export OPENAI_API_KEY="your-key"
```

### 2) 一行代码调用

```go
package main

import (
	"fmt"
	"log"

	"github.com/voocel/litellm"
)

func main() {
	resp, err := litellm.Quick("gpt-4o-mini", "你好，LiteLLM！")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Content)
}
```

### 3) 创建 Client（推荐在项目中使用）

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
			{Role: "user", Content: "用一句话解释 AI。"},
		},
	})
}
```

> 说明
> - `providers` 子包是内部实现细节，使用者只需要引入 `github.com/voocel/litellm`。
> - 模型字符串会原样传给上游 API；智能解析只负责选择 Provider，请优先使用各平台官方模型 ID。

## 核心 API

- `New(opts...)` 创建客户端；不传参数时会从环境变量自动发现可用 Provider。
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

### 推理模型 / Responses API（OpenAI）

```go
resp, err := client.Chat(ctx, &litellm.Request{
	Model: "o3-mini",
	Messages: []litellm.Message{{Role: "user", Content: "逐步算 15*8"}},
	ReasoningEffort:  "medium",
	ReasoningSummary: "auto",
	UseResponsesAPI:  true,
})
_ = resp
```

### 重试与超时

```go
client, _ := litellm.New(
	litellm.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
	litellm.WithRetries(3, 1*time.Second),
	litellm.WithTimeout(60*time.Second),
)
_ = client
```

### 平台特定参数

使用 `Request.Extra` 传递平台特定参数（例如 Qwen/GLM 的 thinking）。参考 `examples/qwen`、`examples/glm`、`examples/bedrock`。

## 自定义 Provider

实现 `litellm.Provider` 并注册即可扩展平台：

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
			Name:            "我的模型",
			MaxOutputTokens: 4096,
			Capabilities:    []litellm.ModelCapability{litellm.CapabilityChat},
		},
	}
}

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

LiteLLM 不会改写模型 ID，只做 Provider 选择，请使用官方模型 ID。

## 配置

用于自动发现的环境变量：

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

## 许可证

Apache License
