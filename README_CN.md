# LiteLLM Go

[English](README.md) | 中文

LiteLLM 是一个小巧、显式、类型化的 Go LLM SDK。根包拥有跨 Provider 的领域模型，具体 Provider 放在 `provider/<name>` 子包。

## 安装

```bash
go get github.com/voocel/litellm
```

## 快速开始

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
			litellm.UserText("用一句话解释 Go interface。"),
		},
		MaxTokens: litellm.IntPtr(120),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
```

`openai.NewClient(cfg, opts...)` 会先创建 provider，再创建 `*litellm.Client`；每个 provider 包都提供。显式两步写法 —— `provider, _ := openai.New(cfg)` 再 `litellm.New(provider, opts...)` —— 完全等价；当你想把同一个 provider 复用到多个 client 时用它。两种写法接受相同的 `ClientOption`。

## 核心模型

消息和响应都由有序 `Block` 表达：

- `TextBlock`
- `ImageBlock`
- `ReasoningBlock`
- `ToolUseBlock`
- `ToolResultBlock`
- `ToolReferenceBlock`

`Response.Blocks` 是规范响应内容；`Text()`、`Reasoning()`、`ToolCalls()` 都只是便利视图。

```go
msgs := []litellm.Message{
	litellm.User(litellm.Text("图里有什么？"), litellm.ImageURL("https://example.com/cat.png")),
}

resp, err := client.Chat(ctx, litellm.Request{Model: "gpt-5.6", Messages: msgs})
_ = resp
_ = err
```

多轮工具调用可以把上一轮响应块原样接回去：

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

`JSONRaw` 会返回 marshal 错误，不会静默生成非法工具参数。`MustJSONRaw` 只建议用于测试数据或允许 panic 的静态示例。

Client 只校验公共模型结构，Provider 负责各自协议约束。历史是否闭合由应用显式检查；导入历史的修复也是独立步骤，不再提供 `WithMessageRepair`：

```go
if err := litellm.ValidateHistory(msgs); err != nil {
    log.Fatal(err)
}

// 仅在应用明确选择修复时调用；原始 msgs 不会被修改。
repaired, warnings := litellm.RepairMessages(msgs, litellm.RepairAll)
_ = repaired
_ = warnings
```

修复返回的 warnings 由应用自行处理。Provider 规范化的 warning 仍通过 `Response.Warnings`、`WarningEvent` 和 `CallObserver.OnEvent` 暴露。修复产生的工具结果只用于标记中断，不代表工具实际执行过。

默认不会保存 Provider 原始响应体。调试时需要显式开启：

```go
client, err := openai.NewClient(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")}, litellm.WithCaptureRawResponse(true))
```

## 流式

流式返回 typed `Event`。
支持显式内容块边界的 Provider 会发出 `ContentStart` / `ContentEnd`：前者包含初始内容，后者可携带最终完整快照（含元数据），不能当作增量重复拼接；快照文本必须与已输出文本一致，否则返回错误和部分响应。`Collect` 自动处理；`StreamText` / `StreamWith` 会交付初始文本和后续增量。
`Stream` 设计为单 goroutine 消费；不要并发调用 `Next`。
如果需要每个事件之间的空闲超时，用 `WithStreamIdleTimeout` 显式开启；默认关闭。
`WithStreamIdleTimeout` 只覆盖通用 `Client.Stream`；OpenAI Responses 原生流用 `openai.Config.StreamIdleTimeout`。
例如：

```go
client, err := openai.NewClient(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")}, litellm.WithStreamIdleTimeout(120*time.Second))
```

```go
stream, err := client.Stream(ctx, litellm.Request{
	Model:    "gpt-5.6",
	Messages: []litellm.Message{litellm.UserText("讲个短笑话。")},
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
		// Provider 原生生命周期或 hosted tool 事件。
	case litellm.DoneEvent:
		return
	}
}
```

聚合流式响应（出错时同时返回部分响应和错误，必须先检查错误）：

```go
resp, err := litellm.Collect(stream)
```

## Retry

默认不重试。`LiteLLMError.Temporary` / `IsTemporaryError` 仅描述故障可能是暂时的，不承诺请求可安全重放。启用重试意味着应用接受重复请求及重复计费的可能；传输层只按配置重试指定 HTTP 状态，不重试网络错误或已开始的响应流。需要时在具体 Provider 上显式开启：

```go
import "github.com/voocel/litellm/retry"

provider, err := openai.New(openai.Config{
	APIKey: os.Getenv("OPENAI_API_KEY"),
	Retry:  retry.DefaultPolicy(),
})
```

Bedrock 的 retry 会在每次 attempt 内部重新签名，用户不需要手动组合 SigV4 transport。

如果需要代理、trace 或自定义底层链路，使用 `Transport` 搭配 `Retry`。完整 `HTTPClient` 是高级逃生口，不能和 `Retry` 同时使用；这种情况下需要用户在自定义 client 内自行配置 retry。

选择规则：

| 场景 | 配置 |
| --- | --- |
| 普通重试 | `Retry: retry.DefaultPolicy()` |
| 重试 + 代理/trace/自定义底层链路 | `Retry` + `Transport` |
| 完全自定义请求执行 | `HTTPClient`，不和 `Retry`/`Transport` 混用 |

`APIKeyFunc` 会在请求创建时解析一次；retry attempt 会复用该请求。如果你使用极短有效期的 Bearer token，请用自定义 `Transport` 或 `HTTPClient` 在更底层注入认证。常规 API key 和默认 retry 窗口不需要关心这个细节。

## 工具调用

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
	Messages:   []litellm.Message{litellm.UserText("巴黎天气？")},
	Tools:      []litellm.Tool{tool},
	ToolChoice: &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto},
})
```

## 结构化输出

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
	Messages:       []litellm.Message{litellm.UserText("生成一个人。")},
	ResponseFormat: format,
})
```

## Thinking

Thinking 必须显式设置。`Thinking == nil` 时 SDK 不发送任何 thinking/reasoning 控制字段。

```go
resp, err := client.Chat(ctx, litellm.Request{
	Model:    "claude-sonnet-5",
	Messages: []litellm.Message{litellm.UserText("解释一下取舍。")},
	MaxTokens: litellm.IntPtr(2048),
	Thinking: &litellm.Thinking{
		Mode:  litellm.ThinkingEnabled,
		Effort: "low",
	},
})
```

稳定的 Provider 约束会在本地校验；模型特有的 effort 和 disable 限制交给官方 API，因此同一 API 代际的新模型无需更新 SDK。
通用 effort 值为 `minimal`、`low`、`medium`、`high`、`xhigh`、`max`，实际支持范围取决于模型。
多 Provider UI 或预检可以用 `client.Capabilities(model)` 或 `litellm.GetCapabilities(provider, model)` 查询稳定能力基线；基线之外的模型特有值仍可发送，并由官方 API 校验。

## OpenAI Responses

设置 `openai.Config.API = openai.APIResponses` 后，通用 `Client.Chat` 和 `Client.Stream` 会通过 Responses API 发送请求，继续使用统一的 `litellm.Request` 和返回类型。默认使用 Chat Completions API。

```go
client, err := openai.NewClient(openai.Config{
	APIKey: os.Getenv("OPENAI_API_KEY"),
	API:    openai.APIResponses,
})
```

需要 hosted tools、conversation ID、`previous_response_id` 等原生字段时，使用 `provider/openai.Provider` 上的 `Responses` 和 `ResponsesStream`：

```go
oai, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
if err != nil {
	log.Fatal(err)
}

resp, err := oai.Responses(ctx, &openai.ResponsesRequest{
	Model: "gpt-5.6",
	Messages: []litellm.Message{
		litellm.UserText("逐步计算 15*8。"),
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

Responses streaming 使用同一套 typed event：

```go
oai, err := openai.New(openai.Config{
	APIKey:            os.Getenv("OPENAI_API_KEY"),
	StreamIdleTimeout: 120 * time.Second,
})

stream, err := oai.ResponsesStream(ctx, &openai.ResponsesRequest{
	Model:    "gpt-5.6",
	Messages: []litellm.Message{litellm.UserText("搜索并总结。")},
})
```

## Provider

Provider 配置属于各自子包。认证不会被强行收口成一个 API key string。

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

示例：

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

当前 provider 子包包括 OpenAI、Anthropic、Gemini、Bedrock、DeepSeek、Qwen、GLM、OpenRouter、MiniMax、Grok、MiMo、Ollama。
各 Provider 的 thinking、reasoning、usage、cache 支持见 [Provider Capabilities](provider-capabilities.md)。

## 模型列表

```go
models, err := client.ListModels(ctx)
```

只有实现了 `ModelLister` 的 Provider 支持该能力，返回字段为 best-effort。

## Provider Options

`Request.ProviderOptions` 是 `map[string]json.RawMessage`，只承载 JSON 数据。通过 `NewProviderOptions` 或 `Set` 在设置时编码，错误直接返回；Client 为执行和 Observer 分别复制 JSON 字节。Provider 在自己的边界解码并校验支持的 key，未知 key 默认报错。

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

`ToolChoice` 也不再接受字符串或协议对象：使用 `&litellm.ToolChoice{Mode: litellm.ToolChoiceAuto}`（也支持 `None` / `Required`），或 `&litellm.ToolChoice{Name: "lookup"}` 指定工具；`nil` 保留 Provider 默认行为。不要在调用期间并发修改传入请求。

## Observer 与 OTel

`Observer.Start` 为每次 Chat/Stream 调用创建独立的 `CallObserver`，包括本地校验失败的调用。Start 收到的是应用传入请求的隔离副本，时机在默认值和校验之前；返回的 context 会依次传给后续 Observer、Provider 和 HTTP 请求。Observer 工厂可并发执行，每次调用独立持有状态。

`OnEvent` 接收已校验的流事件和 WarningEvent（也包含 Chat 的 warning）。`End` 恰好调用一次，并按注册顺序逆序结束；结果包含状态、完整调用耗时、错误以及最终/部分响应。状态分别为 `completed`、`failed`、`canceled`、`closed`。建流成功不代表调用结束；需要持续 Next 到终止或显式 Close，仅取消 context 不会在后台执行回调。

请求、事件和结果均为隔离副本。回调同步执行，核心不 recover panic。应用消费回调的错误由消费函数返回，不改写模型执行结果；关闭尚未完成的流记录为 closed。完成后发生的资源清理错误由 Close 返回，不改写已完成结果。deadline 和 idle timeout 为 failed，主动 context cancellation 为 canceled。

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

可选的 `github.com/voocel/litellm/otel` 模块为每次调用创建 span，并将其 context 传到传输层。它直接读取最终或部分响应，不再维护全局调用 map、锁或重复的流聚合器。默认只记录模型、用量等元数据；显式开启内容捕获后才记录可能包含用户数据和工具参数的消息：

```go
import litellmotel "github.com/voocel/litellm/otel"

observer := litellmotel.New(tracer, litellmotel.WithCaptureContent(true))
client, err := litellm.New(provider, litellm.WithObservers(observer))
```

迁移：删除 `Hook`、`HookFuncs`、`CallMeta` 与 `WithHook(s)`，改用 `ObserverFunc`、`CallObserverFuncs`、`CallInfo`/`CallResult` 和 `WithObservers`，不提供兼容层。开发期间 `otel/go.mod` 通过本地 replace 指向 `..`；发布时应先发布新核心 API，再更新 OTel 的核心依赖版本并移除本地 replace。


## Usage

所有 token 字段均为 `*int`：`nil` 表示未知，`litellm.IntPtr(0)` 表示已知零值。输入用量包含缓存读取和写入，输出用量包含 reasoning；明细是子集，不能再次加到总量上。Anthropic / Bedrock 的适配层会把独立缓存计数并入输入，Gemini 会把 thoughts 并入输出。未上报的明细仍保留 `nil`，OTel 不会把它们记作零。

```go
if resp.Usage.InputTokens != nil {
    fmt.Println(*resp.Usage.InputTokens)
}
```

Pricing 要求已知输入和输出计数；缓存采用不同费率时，对应缓存计数也必须已知，否则返回错误。缓存读写只计费一次；不一致的负数或超出输入总量的缓存计数会报错。未配置缓存费率时沿用普通输入费率。

## Pricing

Pricing 是显式行为。成本计算绝不会隐式联网加载远程定价。

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

## 自定义 Provider

实现很小的 Provider 接口即可：

```go
type Provider interface {
	Name() string
	Chat(context.Context, *litellm.Request) (*litellm.Response, error)
	Stream(context.Context, *litellm.Request) (litellm.Stream, error)
}
```

## 许可证

Apache License
