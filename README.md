# LiteLLM - Go Multi-Platform LLM API Client

**LiteLLM** - Making LLM API calls simple and elegant

[中文](README_CN.md) | English

A clean and elegant Go library for unified access to multiple LLM platforms.

## Key Design Principles

- **Single Entry Point** - Only `litellm.New()` - no confusing choice between multiple APIs
- **Auto-Resolution** - Models automatically resolve to correct providers (gpt-4o → OpenAI, claude → Anthropic)
- **Type-Safe Configuration** - `WithOpenAI()`, `WithAnthropic()` instead of error-prone string-based config
- **Zero Configuration** - Works immediately with environment variables
- **Provider-Agnostic** - Same code works across all AI providers

## Features

- **Simple & Clean** - One-line API calls to any LLM platform
- **Unified Interface** - Same request/response format across all providers
- **Network Resilience** - Automatic retry with exponential backoff and jitter
- **Structured Outputs** - JSON Schema validation with cross-provider support
- **Reasoning Support** - Full support for OpenAI o-series reasoning models
- **Function Calling** - Complete Function Calling support
- **Streaming** - Real-time streaming responses
- **Zero Config** - Auto-discovery from environment variables
- **Extensible** - Easy to add new LLM platforms
- **Type Safe** - Strong typing and comprehensive error handling

## Quick Start

### Installation

```bash
go get github.com/voocel/litellm
```

### One-Line Usage

```go
package main

import (
    "fmt"
    "github.com/voocel/litellm"
)

func main() {
    // Set environment variable: export OPENAI_API_KEY="your-key"
    response, err := litellm.Quick("gpt-4o-mini", "Hello, LiteLLM!")
    if err != nil {
        panic(err)
    }
    fmt.Println(response.Content)
}
```

### Full Configuration

```go
package main

import (
    "context"
    "fmt"
    "github.com/voocel/litellm"
)

func main() {
    // Method 1: Auto-discovery from environment variables
    client := litellm.New()

    // Method 2: Type-safe manual configuration (recommended for production)
    client = litellm.New(
        litellm.WithOpenAI("your-openai-key"),
        litellm.WithAnthropic("your-anthropic-key"),
        litellm.WithGemini("your-gemini-key"),
        litellm.WithQwen("your-dashscope-key"),
        litellm.WithGLM("your-glm-key"),
        litellm.WithOpenRouter("your-openrouter-key"),
        litellm.WithDefaults(2048, 0.8), // Custom defaults
    )

    // Basic chat
    response, err := client.Chat(context.Background(), &litellm.Request{
        Model: "gpt-4o-mini", // Auto-resolves to OpenAI provider
        Messages: []litellm.Message{
            {Role: "user", Content: "Explain artificial intelligence"},
        },
        MaxTokens:   litellm.IntPtr(200),
        Temperature: litellm.Float64Ptr(0.7),
    })

    if err != nil {
        panic(err)
    }

    fmt.Printf("Response: %s\n", response.Content)
    fmt.Printf("Tokens: %d (input: %d, output: %d)\n",
        response.Usage.TotalTokens,
        response.Usage.PromptTokens,
        response.Usage.CompletionTokens)
}
```

## Reasoning Models

Full support for OpenAI o-series reasoning models with both Chat API and Responses API:

```go
response, err := client.Chat(context.Background(), &litellm.Request{
    Model: "o3-mini", // Auto-resolves to OpenAI provider
    Messages: []litellm.Message{
        {Role: "user", Content: "Calculate 15 * 8 step by step"},
    },
    MaxTokens:        litellm.IntPtr(500),
    ReasoningEffort:  "medium",      // "low", "medium", "high"
    ReasoningSummary: "detailed",    // "concise", "detailed", "auto"
    UseResponsesAPI:  true,          // Force Responses API
})

// Access reasoning process
if response.Reasoning != nil {
    fmt.Printf("Reasoning: %s\n", response.Reasoning.Summary)
    fmt.Printf("Reasoning tokens: %d\n", response.Reasoning.TokensUsed)
}
```

## Network Resilience

Built-in automatic retry with exponential backoff for network failures and API errors.

```go
// Default: 3 retries with smart backoff
client := litellm.New(litellm.WithOpenAI("your-api-key"))

// Custom timeout
client := litellm.New(
    litellm.WithOpenAI("your-api-key"),
    litellm.WithTimeout(60*time.Second),
)

// Custom retries
client := litellm.New(
    litellm.WithOpenAI("your-api-key"),
    litellm.WithRetries(5, 2*time.Second), // 5 retries, 2s initial delay
)
```

## Streaming

Real-time streaming with reasoning process display:

```go
stream, err := client.Stream(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini", // Auto-resolves to OpenAI provider
    Messages: []litellm.Message{
        {Role: "user", Content: "Tell me a programming joke"},
    },
})

defer stream.Close()
for {
    chunk, err := stream.Next()
    if err != nil || chunk.Done {
        break
    }

    switch chunk.Type {
    case litellm.ChunkTypeContent:
        fmt.Print(chunk.Content)
    case litellm.ChunkTypeReasoning:
        fmt.Printf("[Thinking: %s]", chunk.Reasoning.Summary)
    }
}
}
```

## Structured Outputs

LiteLLM supports structured JSON outputs with JSON Schema validation, ensuring reliable and predictable responses across all providers.

### Basic JSON Object Output

```go
response, err := client.Chat(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "Generate a person's information"},
    },
    ResponseFormat: litellm.NewResponseFormatJSONObject(),
})

// Response will be valid JSON
fmt.Println(response.Content) // {"name": "John Doe", "age": 30, ...}
```

### JSON Schema with Strict Validation

```go
// Define your data structure
personSchema := map[string]interface{}{
    "type": "object",
    "properties": map[string]interface{}{
        "name": map[string]interface{}{
            "type": "string",
            "description": "Full name",
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

response, err := client.Chat(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "Generate a software engineer's profile"},
    },
    ResponseFormat: litellm.NewResponseFormatJSONSchema(
        "person_profile",
        "A person's professional profile",
        personSchema,
        true, // strict mode
    ),
})

// Parse into your Go struct
type Person struct {
    Name  string `json:"name"`
    Age   int    `json:"age"`
    Email string `json:"email"`
}

var person Person
json.Unmarshal([]byte(response.Content), &person)
```

### Cross-Provider Compatibility

Structured outputs work across all providers with intelligent adaptation:

- **OpenAI**: Native JSON Schema support with strict mode
- **Anthropic**: Prompt engineering with JSON instructions
- **Gemini**: Native response schema support
- **Other providers**: Automatic fallback to prompt-based JSON generation

```go
// Works with any provider
providers := []string{"gpt-4o-mini", "claude-4-sonnet", "gemini-2.5-flash"}

for _, model := range providers {
    response, _ := client.Chat(ctx, &litellm.Request{
        Model: model,
        Messages: []litellm.Message{
            {Role: "user", Content: "Generate user data"},
        },
        ResponseFormat: litellm.NewResponseFormatJSONObject(),
    })
    // All providers return valid JSON
}
```

## Function Calling

### Basic Function Calling
Complete Function Calling support compatible with OpenAI and Anthropic:

```go
tools := []litellm.Tool{
    {
        Type: "function",
        Function: litellm.FunctionSchema{
            Name:        "get_weather",
            Description: "Get weather information for a city",
            Parameters: map[string]interface{}{
                "type": "object",
                "properties": map[string]interface{}{
                    "city": map[string]interface{}{
                        "type":        "string",
                        "description": "City name",
                    },
                },
                "required": []string{"city"},
            },
        },
    },
}

response, err := client.Chat(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "What's the weather in Beijing?"},
    },
    Tools:      tools,
    ToolChoice: "auto",
})

// Handle tool calls
if len(response.ToolCalls) > 0 {
    // Execute function and continue conversation...
}
```

### Advanced Streaming Tool Calls
Real-time streaming with incremental tool call processing:

```go
// Start streaming with tool calls
stream, err := client.Stream(context.Background(), &litellm.Request{
    Model: "gpt-4.1-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "What's the weather like in Tokyo and New York? Use celsius."},
    },
    Tools:      tools,
    ToolChoice: "auto",
})

// Track tool calls with incremental data
toolCalls := make(map[string]*ToolCallBuilder)

defer stream.Close()
for {
    chunk, err := stream.Next()
    if err != nil || chunk.Done {
        break
    }

    switch chunk.Type {
    case litellm.ChunkTypeContent:
        fmt.Print(chunk.Content)

    case litellm.ChunkTypeToolCallDelta:
        // Handle incremental tool call data
        if chunk.ToolCallDelta != nil {
            delta := chunk.ToolCallDelta

            // Create or get tool call builder
            if _, exists := toolCalls[delta.ID]; !exists && delta.ID != "" {
                toolCalls[delta.ID] = &ToolCallBuilder{
                    ID:   delta.ID,
                    Type: delta.Type,
                    Name: delta.FunctionName,
                }
                fmt.Printf("\nTool call started: %s", delta.FunctionName)
            }

            // Accumulate arguments
            if delta.ArgumentsDelta != "" && delta.ID != "" {
                if builder, exists := toolCalls[delta.ID]; exists {
                    builder.Arguments.WriteString(delta.ArgumentsDelta)
                    fmt.Print(".")
                }
            }
        }
    }
}

// Process completed tool calls
for id, builder := range toolCalls {
    fmt.Printf("\nTool: %s(%s)", builder.Name, builder.Arguments.String())
    // Execute the function with the accumulated arguments
}
```

```go
// ToolCallBuilder helps accumulate tool call data
type ToolCallBuilder struct {
    ID        string
    Type      string
    Name      string
    Arguments strings.Builder
}
```

### Reasoning Mode (Qwen3 Thinking)

Qwen3-Coder models support step-by-step reasoning through the `enable_thinking` parameter, providing detailed thinking process for complex coding and mathematical problems:

```go
// Enable reasoning mode for complex problem solving
response, err := client.Chat(ctx, &litellm.Request{
    Model: "qwen3-coder-plus",
    Messages: []litellm.Message{
        {Role: "user", Content: "Write a Python function to implement binary search. Explain your approach step by step."},
    },
    Extra: map[string]interface{}{
        "enable_thinking": true, // Enable Qwen3 reasoning mode
    },
})

if err != nil {
    log.Fatal(err)
}

fmt.Printf("Final Answer: %s\n", response.Content)
if response.Reasoning != nil {
    fmt.Printf("Reasoning Process: %s\n", response.Reasoning.Content)
    fmt.Printf("Reasoning Summary: %s\n", response.Reasoning.Summary)
    fmt.Printf("Reasoning Tokens: %d\n", response.Reasoning.TokensUsed)
}
```

### Reasoning Mode (GLM-4.5 Thinking)

GLM-4.5 models support hybrid reasoning capabilities through the `enable_thinking` parameter, providing step-by-step analysis for complex problems:

```go
// Enable thinking mode for GLM-4.5
response, err := client.Chat(ctx, &litellm.Request{
    Model: "glm-4.5",
    Messages: []litellm.Message{
        {Role: "user", Content: "Design an efficient algorithm to solve the traveling salesman problem and analyze its time complexity."},
    },
    Extra: map[string]interface{}{
        "enable_thinking": true, // Enable GLM-4.5 thinking mode
    },
})

if err != nil {
    log.Fatal(err)
}

fmt.Printf("Final Answer: %s\n", response.Content)
if response.Reasoning != nil {
    fmt.Printf("Reasoning Process: %s\n", response.Reasoning.Content)
    fmt.Printf("Reasoning Summary: %s\n", response.Reasoning.Summary)
    fmt.Printf("Reasoning Tokens: %d\n", response.Reasoning.TokensUsed)
}
```

## Extending New Platforms

Adding new LLM platforms is simple with the custom provider registration system:

```go
// 1. Implement the Provider interface
type MyProvider struct {
    name   string
    config litellm.ProviderConfig
}

func (p *MyProvider) Name() string { return p.name }
func (p *MyProvider) Validate() error { return nil }
func (p *MyProvider) SupportsModel(model string) bool { return true }
func (p *MyProvider) Models() []litellm.ModelInfo {
    return []litellm.ModelInfo{
        {ID: "my-model", Provider: "myprovider", Name: "My Model", MaxTokens: 4096},
    }
}

func (p *MyProvider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
    // Implement your API call logic here
    return &litellm.Response{
        Content:  "Hello from my provider!",
        Model:    req.Model,
        Provider: p.name,
        Usage:    litellm.Usage{TotalTokens: 10},
    }, nil
}

func (p *MyProvider) Stream(ctx context.Context, req *litellm.Request) (litellm.StreamReader, error) {
    // Implement streaming if needed
    return nil, fmt.Errorf("streaming not implemented")
}

// 2. Create a factory function
func NewMyProvider(config litellm.ProviderConfig) litellm.Provider {
    return &MyProvider{name: "myprovider", config: config}
}

// 3. Register the provider
func init() {
    litellm.RegisterProvider("myprovider", NewMyProvider)
}

// 4. Use it
client := litellm.New(
    litellm.WithProviderConfig("myprovider", litellm.ProviderConfig{
        APIKey: "your-api-key",
    }),
)
response, _ := client.Chat(ctx, &litellm.Request{
    Model: "my-model",
    Messages: []litellm.Message{{Role: "user", Content: "Hello"}},
})
```

### Provider Discovery

```go
// List all available providers
providers := litellm.ListRegisteredProviders()
fmt.Printf("Available providers: %v\n", providers)

// Check if a provider is registered
if litellm.IsProviderRegistered("myprovider") {
    fmt.Println("Custom provider is available!")
}
```

## Supported Platforms

### OpenAI
- GPT-5, GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-mano
- o3, o3-mini, o4-mini (reasoning models)
- Chat Completions API & Responses API
- Function Calling, Vision, Streaming


#### Usage tips for GPT-5
- For deeper reasoning, set `ReasoningEffort` and/or `ReasoningSummary`. LiteLLM will automatically switch to the Responses API and use `MaxCompletionTokens` for better reasoning behavior.
- Consider increasing `MaxCompletionTokens` to cover both reasoning and final answer tokens.
- Ensure your API key has access to `gpt-5`; otherwise, requests may fail. You can also try routing via OpenRouter (`openai/gpt-5`).

### Anthropic
- Claude 3.7 Sonnet, Claude 4 Sonnet, Claude 4 Opus
- Function Calling, Vision, Streaming

### Google Gemini
- Gemini 2.5 Pro, Gemini 2.5 Flash
- Function Calling, Vision, Streaming
- Large context window

### DeepSeek
- DeepSeek Chat, DeepSeek Reasoner
- Function Calling, Code Generation, Reasoning
- Large context window

### Qwen (Alibaba Cloud DashScope)
- Qwen3-Coder-Plus, Qwen3-Coder-Flash (with thinking mode support)
- Qwen3-Coder-480B-A35B-Instruct, Qwen3-Coder-30B-A3B-Instruct (open source models)
- Function Calling, Code Generation, Reasoning (step-by-step thinking via `enable_thinking`), Large context window (up to 1M tokens)
- OpenAI-compatible API through DashScope

### GLM (智谱AI)
- GLM-4.5 (355B-A32B flagship model with hybrid reasoning capabilities)
- GLM-4.5-Air (106B-A12B lightweight version), GLM-4.5-Flash (fast version)
- GLM-4, GLM-4-Flash, GLM-4-Air, GLM-4-AirX (previous generation models)
- Function Calling, Code Generation, Reasoning (thinking mode), Large context window (128K tokens)
- OpenAI-compatible API through Zhipu AI Open Platform

### OpenRouter
- Access to 200+ models from multiple providers
- OpenAI, Anthropic, Google, Meta, and more
- Unified API for all supported models
- Reasoning models support

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export DEEPSEEK_API_KEY="sk-..."
export QWEN_API_KEY="sk-..."  # For Qwen models
export GLM_API_KEY="your-glm-key"  # For GLM models
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Code Configuration (Recommended)
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

## API Reference

### Core Types
```go
type Request struct {
    Model            string          `json:"model"`                 // Model name
    Messages         []Message       `json:"messages"`              // Conversation messages
    MaxTokens        *int            `json:"max_tokens,omitempty"`  // Max tokens to generate
    Temperature      *float64        `json:"temperature,omitempty"` // Sampling temperature
    Tools            []Tool          `json:"tools,omitempty"`       // Available tools
    ResponseFormat   *ResponseFormat `json:"response_format,omitempty"` // Response format
    ReasoningEffort  string          `json:"reasoning_effort,omitempty"`  // Reasoning effort
    ReasoningSummary string          `json:"reasoning_summary,omitempty"` // Reasoning summary
}

type Response struct {
    Content   string         `json:"content"`              // Generated content
    ToolCalls []ToolCall     `json:"tool_calls,omitempty"` // Tool calls
    Usage     Usage          `json:"usage"`                // Token usage
    Reasoning *ReasoningData `json:"reasoning,omitempty"`  // Reasoning data
}

type ResponseFormat struct {
    Type       string      `json:"type"`                 // "text", "json_object", "json_schema"
    JSONSchema *JSONSchema `json:"json_schema,omitempty"` // JSON schema for structured output
}

type JSONSchema struct {
    Name        string `json:"name"`                  // Schema name
    Description string `json:"description,omitempty"` // Schema description
    Schema      any    `json:"schema"`                // JSON schema definition
    Strict      *bool  `json:"strict,omitempty"`      // Whether to enforce strict adherence
}
```

### Main Methods
```go
func Quick(model, message string) (*Response, error)
func New(opts ...ClientOption) *Client
func (c *Client) Chat(ctx context.Context, req *Request) (*Response, error)
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error)
```

## License

Apache License