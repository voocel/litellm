# LiteLLM Go

[English](README.md) | 中文

LiteLLM 是一个小巧、显式、类型化的 Go LLM SDK。根包拥有跨 Provider 的领域模型，具体 Provider 放在 `provider/<name>` 子包。

SDK 只做结构映射：不推断模型支持什么，不在本地校验厂商取值，也不改写你的输入。由厂商 API 裁决，其错误以类型化的 `*litellm.Error` 返回。

## 安装

```bash
go get github.com/voocel/litellm
```

需要 Go 1.26 或更高版本。

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

Provider 可在多个 Client 间共享。`litellm.New` 接受 `WithObservers`、`WithStreamIdleTimeout`、`WithCaptureRawResponse` 等 `ClientOption`。

## 核心模型

消息与响应由有序的 `Block` 组成：`TextBlock`、`ImageBlock`、`ReasoningBlock`、`ToolUseBlock`、`ToolResultBlock`、`ToolReferenceBlock`。`Response.Blocks` 是权威内容；`Text()`、`Reasoning()`、`ToolCalls()` 只是视图。

```go
msgs := []litellm.Message{
	litellm.User(litellm.Text("What is in this image?"), litellm.ImageURL("https://example.com/cat.png")),
}
```

多轮工具调用时，直接追加上一轮的响应块，持久化历史时保持块完整：

```go
msgs = append(msgs,
	litellm.Assistant(resp.Blocks...),
	litellm.ToolResultText("call_1", `{"ok":true}`),
)
```

推理签名、item id 等厂商需要原样取回的数据放在块的 `State` 中，只回传给产生它的 Provider，因此历史可以在 Provider 之间切换（[详见](providers.md#replay-state)）。Client 只校验消息结构，从不改写历史。工具调用配对与修复属于会话策略，由持有会话的上层负责。

原始响应体仅在 `litellm.WithCaptureRawResponse(true)` 时保留。

## 流式

流是一串块。`BlockStart` 打开 `Index` 处的块，`Index` 即它在 `Response.Blocks` 中的位置；`TextDelta`、`ReasoningDelta`、`ToolUseDelta` 追加内容；`BlockEnd` 以完整块（含 `State` 等后到的元数据）关闭它。块可以交错，但都在 `DoneEvent` 之前结束。`UsageEvent`、`WarningEvent` 与 `ProviderEvent`（无类型对应的原生事件）承载其余信息。

`Handle` 聚合流并把每个事件交给回调；`Collect` 只聚合。失败时两者都返回部分响应和错误。

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

`Client.Stream` 在读取时即完成聚合，因此即便先用 `Next` 读过部分事件，`Handle` / `Collect` 仍返回完整响应。流只能由一个 goroutine 消费。`WithStreamIdleTimeout` 可设置逐事件空闲超时，默认关闭。

## 工具

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

`ToolChoice` 取 `Mode`（`Auto`、`None`、`Required`）或工具 `Name`；nil 表示沿用厂商默认。

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
	Messages:       []litellm.Message{litellm.UserText("Generate a person.")},
	ResponseFormat: format,
})
```

## Thinking

`Thinking == nil` 不发送任何 thinking 字段，保持厂商默认。否则零值 `Mode` 即开启，`ThinkingDisabled` 关闭；`Effort` 与 `BudgetTokens` 原样发送；`IncludeOutput` 在厂商可选时请求返回推理文本。

```go
resp, err := client.Chat(ctx, litellm.Request{
	Model:     "claude-sonnet-5",
	Messages:  []litellm.Message{litellm.UserText("Explain the tradeoffs.")},
	MaxTokens: new(2048),
	Thinking:  &litellm.Thinking{Effort: "low"},
})
```

模型接受哪些取值由厂商决定。各 Provider 的确切线上映射见 [providers.md](providers.md)。

## 提示缓存

在块上标记缓存断点：截至并包含该块的提示前缀可被缓存。`TTL` 原样传递（`litellm.CacheTTL5m`、`litellm.CacheTTL1h`，或留空使用厂商默认）。断点只是提示，没有对应字段的 Provider 会丢弃。

```go
litellm.User(litellm.TextBlock{Text: longDocument, Cache: &litellm.CacheControl{TTL: litellm.CacheTTL1h}})
```

## Provider Options

`Request.ProviderOptions` 承载原生线上字段：每个键是厂商请求体的顶层字段，值为 JSON。键按 Provider 的清单检查，未知键报错。若键与适配器生成的字段同名：对象合并进去，数组追加进去，其他冲突报错。

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

| 包 | API |
| --- | --- |
| `provider/openai` | OpenAI Chat Completions（默认）或 Responses |
| `provider/anthropic` | Anthropic Messages |
| `provider/gemini` | Gemini `generateContent` |
| `provider/bedrock` | Amazon Bedrock Converse（SigV4） |
| `provider/deepseek`、`glm`、`grok`、`mimo`、`minimax`、`ollama`、`openrouter`、`qwen` | 各厂商的 Chat Completions 方言 |
| `provider/compat` | 其他任意 OpenAI 兼容端点（vLLM、LM Studio、网关） |

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

从配置里选择 provider 的应用可以按名字构造，共享设置放在 `provider.Config`：

```go
import "github.com/voocel/litellm/provider"

provider, err := provider.New("anthropic", provider.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
names := provider.Names() // "anthropic"、"bedrock"、"compat"……
```

`openai` 只讲官方协议。`compat` 用于其他任意 OpenAI 兼容服务，provider option 不检查、原样透传，因为它无从知道服务端字段。

`client.Capabilities()` 报告适配器能表达什么：是否发送 `Thinking`、`ThinkingDisabled`、`Effort`、`BudgetTokens`，是否必须设置 `MaxTokens`，以及可接受的选项键。它按 Provider 静态固定（自定义 Provider 未声明时 `ok` 为 false）；模型是否接受仍由厂商裁决。

### OpenAI Responses

设置 `openai.Config.API = openai.APIResponses`，即可让 `Chat` 与 `Stream` 走 Responses API，请求与响应类型不变。Responses 原生字段通过 provider option 传入；使用另一种 API 的选项会报错。

```go
provider, err := openai.New(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY"), API: openai.APIResponses})

options, err := litellm.NewProviderOptions(map[string]any{
	openai.ProviderOptionPreviousResponseID: "resp_123",
	openai.ProviderOptionTools:              []any{map[string]any{"type": "web_search"}},
})
```

## 模型列表

```go
models, err := client.ListModels(ctx)
```

仅实现了 `ModelLister` 的 Provider 可用，返回字段尽力而为。

## 错误

HTTP 失败与流内错误按同一规则分类；用 `Is*` 判断，不要匹配消息文本：

```go
switch {
case litellm.IsContextOverflowError(err):
	// 压缩历史后重发
case litellm.IsRateLimitError(err), litellm.IsOverloadedError(err):
	time.Sleep(litellm.RetryAfter(err)) // 厂商未给 Retry-After 时为 0
case litellm.IsContentFilterError(err):
	// 不要重试
}
```

消息格式为 `provider: code: message`。即使代理把状态码改写，上下文超限与内容过滤仍按厂商错误码和消息识别，且永不标记为临时错误。

## 重试

Provider 从不重试。`Error.Temporary` 只表示失败可能是暂时的，不代表可以重放：厂商可能已经处理并计费。需要时，包装传给 Provider 的 HTTP 客户端：

```go
import "github.com/voocel/litellm/retry"

provider, err := openai.New(openai.Config{
	APIKey:     os.Getenv("OPENAI_API_KEY"),
	HTTPClient: retry.NewHTTPClient(nil, retry.DefaultPolicy()),
})
```

- 重试：完整的 408、429、500、502、503、504、529 响应，响应体表明额度耗尽、鉴权失败、内容过滤或上下文超限的除外。
- 不重试（原样返回响应或错误）：网络失败、中断的流、请求体无法重发的请求；开启 `RespectRetryAfter`（`DefaultPolicy` 默认开启）时，`Retry-After` 超过 `MaxRetryAfter`（默认 60 秒）的响应。
- Bedrock：重试沿用已签名的 SigV4 请求，签名五分钟内有效。

## Observer 与 OTel

`Observer` 为每次 Chat/Stream 调用（包括本地校验失败）启动一个 `CallObserver`。`CallInfo.Request` 是调用方请求的快照，由所有 Observer 共享。`OnEvent` 实时接收流事件（以及 Chat 的警告）；`End` 只运行一次，带状态（`completed`、`failed`、`canceled`、`closed`）、耗时、错误与最终或部分响应。流必须消费到结束或显式 Close。

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

可选模块 `github.com/voocel/litellm/otel` 为每次调用创建符合 GenAI 语义约定的 span，并把上下文传到传输层。内容采集默认关闭；消息可能含用户数据，需显式开启：

```go
import litellmotel "github.com/voocel/litellm/otel"

observer := litellmotel.New(tracer, litellmotel.WithCaptureContent(true))
```

## 用量与模型目录

Token 计数为 `*int`：nil 表示未知，`new(0)` 表示已知为零。`Input()`、`Output()`、`Total()`、`Reasoning()`、`CacheRead()`、`CacheWrite()` 返回 `(count, known)`。输入含缓存读写，输出含推理，明细计数是子集。

模型目录来自 LiteLLM 的模型表，包含上下文窗口、输出上限、是否支持推理和价格，从不隐式加载远程数据：

```go
import (
	"fmt"

	"github.com/voocel/litellm/catalog"
)

var models catalog.Catalog
if err := models.LoadFromURL(ctx, catalog.DefaultURL); err != nil {
	return err
}

model, ok := models.Get("anthropic/claude-sonnet-4-5") // 精确匹配模型表的键
if !ok {
	return fmt.Errorf("模型未收录")
}
if model.Pricing == nil {
	return fmt.Errorf("模型未提供价格")
}

cost, err := model.Pricing.Cost(resp.Usage)
if err != nil {
	return err
}
```

名称即模型表的键，厂商前缀沿用 LiteLLM 的 provider 名，例如 `xai/`、`zai/`、`dashscope/`，而不是 `provider.Names()`。目录不做名称转换，因为同一模型可能在多个站点收录且价格不同；需要计费的模型请在应用配置里记下它的目录名。内置 Provider 可用 `provider.CatalogName(name, model)` 得到目录名，例如 `grok` 对应 `xai/grok-4`。

`Get` 精确匹配完整键。`vendor/model` 不存在时，即使存在 `model`，也返回 `ok == false`。

`ok == false` 表示模型未收录；`Pricing == nil` 表示价格未知；非 nil 的 `Pricing` 中费率为零表示对应用量免费。

`Model.Reasoning` 为 `*bool`：nil 表示未知，false 表示不支持，true 表示支持。Token 上限为零表示未知。`Set` 和模型表加载都会校验名称、上限和费率；加载失败保留原目录。Provider 不会依据目录改写请求。

## 自定义 Provider

实现 Provider 接口即可；`CapabilityProvider` 与 `ModelLister` 可选：

```go
type Provider interface {
	Name() string
	Chat(context.Context, *litellm.Request) (*litellm.Response, error)
	Stream(context.Context, *litellm.Request) (litellm.Stream, error)
}
```

## 许可证

Apache License
