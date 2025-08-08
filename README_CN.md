# LiteLLM - Go 多平台 LLM API 客户端

**LiteLLM** - 让 LLM API 调用变得简单优雅

[English](README.md) | 中文

一个简洁优雅的 Go 语言库，用于统一访问多个 LLM 平台。

## 特性

- **简洁易用** - 一行代码调用任意 LLM 平台
- **统一接口** - 所有平台使用相同的请求/响应格式
- **结构化输出** - JSON Schema 验证，跨平台支持
- **推理支持** - 完整支持 OpenAI o 系列推理模型
- **工具调用** - 完整的 Function Calling 支持
- **流式处理** - 实时流式响应
- **零配置** - 环境变量自动发现
- **易扩展** - 轻松添加新的 LLM 平台
- **类型安全** - 强类型定义和完善的错误处理

## 快速开始

### 安装

```bash
go get github.com/voocel/litellm
```

### 一行代码使用

```go
package main

import (
    "fmt"
    "github.com/voocel/litellm"
)

func main() {
    // 设置环境变量: export OPENAI_API_KEY="your-key"
    response, err := litellm.Quick("gpt-4o-mini", "你好，LiteLLM！")
    if err != nil {
        panic(err)
    }
    fmt.Println(response.Content)
}
```

### 完整配置

```go
package main

import (
    "context"
    "fmt"
    "github.com/voocel/litellm"
)

func main() {
    // 方式1: 环境变量自动发现
    client := litellm.New()
    
    // 方式2: 手动配置 (生产环境推荐)
    client = litellm.New(
		litellm.WithOpenAI("your-openai-key"),
		litellm.WithAnthropic("your-anthropic-key"),
		litellm.WithGemini("your-gemini-key"),
		litellm.WithOpenRouter("your-openrouter-key"),
		litellm.WithDefaults(2048, 0.8), // 自定义默认参数
    )
    
    // 基础聊天
    response, err := client.Complete(context.Background(), &litellm.Request{
        Model: "gpt-4o-mini",
        Messages: []litellm.Message{
            {Role: "user", Content: "解释什么是人工智能"},
        },
        MaxTokens:   litellm.IntPtr(200),
        Temperature: litellm.Float64Ptr(0.7),
    })
    
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("回答: %s\n", response.Content)
    fmt.Printf("Tokens: %d (输入: %d, 输出: %d)\n", 
        response.Usage.TotalTokens, 
        response.Usage.PromptTokens, 
        response.Usage.CompletionTokens)
}
```

## 结构化输出

LiteLLM 支持结构化 JSON 输出和 JSON Schema 验证，确保跨所有平台获得可靠且可预测的响应。

### 基础 JSON 对象输出

```go
response, err := client.Complete(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "生成一个人的信息"},
    },
    ResponseFormat: litellm.NewResponseFormatJSONObject(),
})

// 响应将是有效的 JSON
fmt.Println(response.Content) // {"name": "张三", "age": 30, ...}
```

### JSON Schema 严格验证

```go
// 定义数据结构
personSchema := map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "name": map[string]interface{}{
            "type": "string",
            "description": "姓名",
        },
        "age": map[string]interface{}{
            "type": "integer",
            "minimum": 0,
            "maximum": 150,
        },
        "email": map[string]interface{}{
            "type": "string",
            "format": "email",
        },
    },
    "required": []string{"name", "age", "email"},
}

response, err := client.Complete(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "生成一个软件工程师的档案"},
    },
    ResponseFormat: litellm.NewResponseFormatJSONSchema(
        "person_profile",
        "个人职业档案信息",
        personSchema,
        true, // 严格模式
    ),
})

// 解析为 Go 结构体
type Person struct {
    Name  string `json:"name"`
    Age   int    `json:"age"`
    Email string `json:"email"`
}

var person Person
json.Unmarshal([]byte(response.Content), &person)
```

### 跨平台兼容性

结构化输出在所有平台上都能工作，具有智能适配：

- **OpenAI**: 原生 JSON Schema 支持，带严格模式
- **Anthropic**: 通过提示工程实现 JSON 指令
- **Gemini**: 原生响应 schema 支持
- **其他平台**: 自动降级到基于提示的 JSON 生成

```go
// 适用于任何平台
providers := []string{"gpt-4o-mini", "claude-4-sonnet", "gemini-2.5-flash"}

for _, model := range providers {
    response, _ := client.Complete(ctx, &litellm.Request{
        Model: model,
        Messages: []litellm.Message{
            {Role: "user", Content: "生成用户数据"},
        },
        ResponseFormat: litellm.NewResponseFormatJSONObject(),
    })
    // 所有平台都返回有效的 JSON
}
```

## 推理模型

完整支持 OpenAI o 系列推理模型，包括 Chat API 和 Responses API：

```go
response, err := client.Complete(context.Background(), &litellm.Request{
    Model: "o3-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "逐步计算 15 * 8"},
    },
    MaxTokens:        litellm.IntPtr(500),
    ReasoningEffort:  "medium",      // "low", "medium", "high"
    ReasoningSummary: "detailed",    // "concise", "detailed", "auto"
    UseResponsesAPI:  true,          // 强制使用 Responses API
})

// 获取推理过程
if response.Reasoning != nil {
    fmt.Printf("推理过程: %s\n", response.Reasoning.Summary)
    fmt.Printf("推理 tokens: %d\n", response.Reasoning.TokensUsed)
}
```

## 流式处理

支持实时流式响应和推理过程展示：

```go
stream, err := client.Stream(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "讲一个编程笑话"},
    },
})

defer stream.Close()
for {
    chunk, err := stream.Read()
    if err != nil || chunk.Done {
        break
    }
    
    switch chunk.Type {
    case litellm.ChunkTypeContent:
        fmt.Print(chunk.Content)
    case litellm.ChunkTypeReasoning:
        fmt.Printf("[思考: %s]", chunk.Reasoning.Summary)
    }
}
```

### 思考模式 (Qwen3 Thinking)

Qwen3-Coder 模型支持通过 `enable_thinking` 参数启用逐步推理，为复杂的编程和数学问题提供详细的思考过程：

```go
// 启用思考模式进行复杂问题求解
response, err := client.Complete(ctx, &litellm.Request{
    Model: "qwen3-coder-plus",
    Messages: []litellm.Message{
        {Role: "user", Content: "编写一个 Python 函数实现二分查找算法。请逐步解释你的方法。"},
    },
    Extra: map[string]interface{}{
        "enable_thinking": true, // 启用 Qwen3 思考模式
    },
})

if err != nil {
    log.Fatal(err)
}

fmt.Printf("最终答案: %s\n", response.Content)
if response.Reasoning != nil {
    fmt.Printf("推理过程: %s\n", response.Reasoning.Content)
    fmt.Printf("推理摘要: %s\n", response.Reasoning.Summary)
    fmt.Printf("推理 Tokens: %d\n", response.Reasoning.TokensUsed)
}
```

### 思考模式 (GLM-4.5 Thinking)

GLM-4.5 模型支持通过 `enable_thinking` 参数启用混合推理能力，为复杂问题提供逐步分析：

```go
// 启用 GLM-4.5 思考模式
response, err := client.Complete(ctx, &litellm.Request{
    Model: "glm-4.5",
    Messages: []litellm.Message{
        {Role: "user", Content: "设计一个高效的算法来解决旅行商问题，并分析其时间复杂度。"},
    },
    Extra: map[string]interface{}{
        "enable_thinking": true, // 启用 GLM-4.5 思考模式
    },
})

if err != nil {
    log.Fatal(err)
}

fmt.Printf("最终答案: %s\n", response.Content)
if response.Reasoning != nil {
    fmt.Printf("推理过程: %s\n", response.Reasoning.Content)
    fmt.Printf("推理摘要: %s\n", response.Reasoning.Summary)
    fmt.Printf("推理 Tokens: %d\n", response.Reasoning.TokensUsed)
}
```

## 工具调用 (Function Calling)

### 基础工具调用
完整支持 Function Calling，兼容 OpenAI 和 Anthropic：

```go
tools := []litellm.Tool{
    {
        Type: "function",
        Function: litellm.FunctionSchema{
            Name:        "get_weather",
            Description: "获取城市天气信息",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "city": map[string]interface{}{
                        "type":        "string",
                        "description": "城市名称",
                    },
                },
                "required": []string{"city"},
            },
        },
    },
}

response, err := client.Complete(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "北京天气怎么样？"},
    },
    Tools:      tools,
    ToolChoice: "auto",
})

// 处理工具调用
if len(response.ToolCalls) > 0 {
    // 执行函数并继续对话...
}
```

### 高级流式工具调用
实时流式处理，支持增量工具调用数据处理：

```go
// 启动流式工具调用
stream, err := client.Stream(context.Background(), &litellm.Request{
    Model: "gpt-4.1-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "东京和纽约的天气怎么样？使用摄氏度。"},
    },
    Tools:      tools,
    ToolChoice: "auto",
})

// 使用增量数据跟踪工具调用
toolCalls := make(map[string]*ToolCallBuilder)

defer stream.Close()
for {
    chunk, err := stream.Read()
    if err != nil || chunk.Done {
        break
    }

    switch chunk.Type {
    case litellm.ChunkTypeContent:
        fmt.Print(chunk.Content)

    case litellm.ChunkTypeToolCallDelta:
        // 处理增量工具调用数据
        if chunk.ToolCallDelta != nil {
            delta := chunk.ToolCallDelta

            // 创建或获取工具调用构建器
            if _, exists := toolCalls[delta.ID]; !exists && delta.ID != "" {
                toolCalls[delta.ID] = &ToolCallBuilder{
                    ID:   delta.ID,
                    Type: delta.Type,
                    Name: delta.FunctionName,
                }
                fmt.Printf("\n工具调用开始: %s", delta.FunctionName)
            }

            // 累积参数
            if delta.ArgumentsDelta != "" && delta.ID != "" {
                if builder, exists := toolCalls[delta.ID]; exists {
                    builder.Arguments.WriteString(delta.ArgumentsDelta)
                    fmt.Print(".")
                }
            }
        }
    }
}

// 处理完成的工具调用
for id, builder := range toolCalls {
    fmt.Printf("\n工具调用: %s(%s)", builder.Name, builder.Arguments.String())
    // 使用累积的参数执行函数
}
```

```go
// ToolCallBuilder 帮助累积工具调用数据
type ToolCallBuilder struct {
    ID        string
    Type      string
    Name      string
    Arguments strings.Builder
}
```

## 扩展新平台

添加新的 LLM 平台非常简单：

```go
// 实现 Provider 接口
type MyProvider struct {
    *litellm.BaseProvider
}

func (p *MyProvider) Complete(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
    // 实现 API 调用逻辑
    return &litellm.Response{
        Content:  "hi！",
        Model:    req.Model,
        Provider: "myprovider",
        Usage:    litellm.Usage{TotalTokens: 10},
    }, nil
}

// 注册 provider
func init() {
    litellm.RegisterProvider("myprovider", NewMyProvider)
}

// 使用
client := litellm.New()
response, _ := client.Complete(ctx, &litellm.Request{
    Model: "my-model",
    Messages: []litellm.Message{{Role: "user", Content: "你好"}},
})
```

## 支持的平台

### OpenAI
- GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-mano
- o3, o3-mini, o4-mini (推理模型)
- Chat Completions API 和 Responses API
- Function Calling, Vision, 流式处理

### Anthropic
- Claude 3.7 Sonnet, Claude 4 Sonnet, Claude 4 Opus
- Function Calling, Vision, 流式处理

### Google Gemini
- Gemini 2.5 Pro, Gemini 2.5 Flash
- Function Calling, Vision, 流式处理
- 超大上下文窗口

### DeepSeek
- DeepSeek Chat, DeepSeek Reasoner
- Function Calling, 代码生成, 推理能力
- 超大上下文窗口

### Qwen (阿里云 DashScope)
- Qwen3-Coder-Plus, Qwen3-Coder-Flash (支持思考模式)
- Qwen3-Coder-480B-A35B-Instruct, Qwen3-Coder-30B-A3B-Instruct (开源模型)
- Function Calling, 代码生成, 推理能力 (通过 `enable_thinking` 参数实现逐步思考), 超大上下文窗口 (最高 1M tokens)
- 通过 DashScope 提供 OpenAI 兼容 API

### GLM (智谱AI)
- GLM-4.5 (355B-A32B 旗舰模型，具备混合推理能力)
- GLM-4.5-Air (106B-A12B 轻量版), GLM-4.5-Flash (快速版)
- GLM-4, GLM-4-Flash, GLM-4-Air, GLM-4-AirX (上一代模型)
- Function Calling, 代码生成, 推理能力 (思考模式), 大上下文窗口 (128K tokens)
- 通过智谱AI开放平台提供 OpenAI 兼容 API

### OpenRouter
- 访问来自多个提供商的 200+ 模型
- OpenAI, Anthropic, Google, Meta 等
- 所有支持模型的统一 API
- 推理模型支持

## 配置

### 环境变量
```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export QWEN_API_KEY="sk-..."
export GLM_API_KEY="your-glm-key"  # GLM 模型
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 代码配置 (推荐)
```go
client := litellm.New(
    litellm.WithOpenAI("your-openai-key"),
    litellm.WithAnthropic("your-anthropic-key"),
    litellm.WithGemini("your-gemini-key"),
    litellm.WithDeepSeek("your-deepseek-key"),
    litellm.WithQwen("your-qwen-key"),
    litellm.WithGLM("your-glm-key"),
    litellm.WithOpenRouter("your-openrouter-key"),
    litellm.WithDefaults(2048, 0.8),
)
```

## API 参考

### 核心类型
```go
type Request struct {
    Model            string          `json:"model"`                 // 模型名称
    Messages         []Message       `json:"messages"`              // 对话消息
    MaxTokens        *int            `json:"max_tokens,omitempty"`  // 最大token数
    Temperature      *float64        `json:"temperature,omitempty"` // 采样温度
    Tools            []Tool          `json:"tools,omitempty"`       // 可用工具
    ResponseFormat   *ResponseFormat `json:"response_format,omitempty"` // 响应格式
    ReasoningEffort  string          `json:"reasoning_effort,omitempty"`  // 推理强度
    ReasoningSummary string          `json:"reasoning_summary,omitempty"` // 推理摘要
}

type Response struct {
    Content   string         `json:"content"`              // 生成内容
    ToolCalls []ToolCall     `json:"tool_calls,omitempty"` // 工具调用
    Usage     Usage          `json:"usage"`                // Token使用统计
    Reasoning *ReasoningData `json:"reasoning,omitempty"`  // 推理数据
}

type ResponseFormat struct {
    Type       string      `json:"type"`                 // "text", "json_object", "json_schema"
    JSONSchema *JSONSchema `json:"json_schema,omitempty"` // 结构化输出的 JSON schema
}

type JSONSchema struct {
    Name        string `json:"name"`                  // Schema 名称
    Description string `json:"description,omitempty"` // Schema 描述
    Schema      any    `json:"schema"`                // JSON schema 定义
    Strict      *bool  `json:"strict,omitempty"`      // 是否强制严格遵守
}
```

### 主要方法
```go
func Quick(model, message string) (*Response, error)
func New(opts ...ClientOption) *Client
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error)
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error)
```

## 许可证

Apache License
