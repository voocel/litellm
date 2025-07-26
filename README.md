# LiteLLM - Go Multi-Platform LLM API Client

[中文](README_CN.md) | English

A clean and elegant Go library for unified access to multiple LLM platforms.

## Features

- **Simple & Clean** - One-line API calls to any LLM platform
- **Unified Interface** - Same request/response format across all providers
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
    
    // Method 2: Manual configuration (recommended for production)
    client = litellm.New(
		litellm.WithOpenAI("your-openai-key"),
		litellm.WithAnthropic("your-anthropic-key"),
		litellm.WithGemini("your-gemini-key"),
		litellm.WithDefaults(2048, 0.8), // Custom defaults
    )
    
    // Basic chat
    response, err := client.Complete(context.Background(), &litellm.Request{
        Model: "gpt-4o-mini",
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
response, err := client.Complete(context.Background(), &litellm.Request{
    Model: "o3-mini",
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

## Streaming

Real-time streaming with reasoning process display:

```go
stream, err := client.Stream(context.Background(), &litellm.Request{
    Model: "gpt-4o-mini",
    Messages: []litellm.Message{
        {Role: "user", Content: "Tell me a programming joke"},
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
        fmt.Printf("[Thinking: %s]", chunk.Reasoning.Summary)
    }
}
```

## Function Calling

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

response, err := client.Complete(context.Background(), &litellm.Request{
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

## Extending New Platforms

Adding new LLM platforms is simple:

```go
// Implement Provider interface
type MyProvider struct {
    *litellm.BaseProvider
}

func (p *MyProvider) Complete(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
    // Implement API call logic
    return &litellm.Response{
        Content:  "Hello from my provider!",
        Model:    req.Model,
        Provider: "myprovider",
        Usage:    litellm.Usage{TotalTokens: 10},
    }, nil
}

// Register provider
func init() {
    litellm.RegisterProvider("myprovider", NewMyProvider)
}

// Use it
client := litellm.New()
response, _ := client.Complete(ctx, &litellm.Request{
    Model: "my-model",
    Messages: []litellm.Message{{Role: "user", Content: "Hello"}},
})
```

## Supported Platforms

### OpenAI
- GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-mano
- o3, o3-mini, o4-mini (reasoning models)
- Chat Completions API & Responses API
- Function Calling, Vision, Streaming

### Anthropic
- Claude 3.7 Sonnet, Claude 4 Sonnet, Claude 4 Opus
- Function Calling, Vision, Streaming

### Google Gemini
- Gemini 2.5 Pro, Gemini 2.5 Flash
- Function Calling, Vision, Streaming
- Large context window

## Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="sk-proj-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
```

### Code Configuration (Recommended)
```go
client := litellm.New(
    litellm.WithOpenAI("your-openai-key"),
    litellm.WithAnthropic("your-anthropic-key"),
    litellm.WithGemini("your-gemini-key"),
    litellm.WithDefaults(2048, 0.8),
)
```

## API Reference

### Core Types
```go
type Request struct {
    Model            string    `json:"model"`                 // model namel
    Messages         []Message `json:"messages"`              // Conversation messages
    MaxTokens        *int      `json:"max_tokens,omitempty"`  // Max tokens to generate
    Temperature      *float64  `json:"temperature,omitempty"` // Sampling temperature
    Tools            []Tool    `json:"tools,omitempty"`       // Available tools
    ReasoningEffort  string    `json:"reasoning_effort,omitempty"`  // Reasoning effort
    ReasoningSummary string    `json:"reasoning_summary,omitempty"` // Reasoning summary
}

type Response struct {
    Content   string         `json:"content"`              // Generated content
    ToolCalls []ToolCall     `json:"tool_calls,omitempty"` // Tool calls
    Usage     Usage          `json:"usage"`                // Token usage
    Reasoning *ReasoningData `json:"reasoning,omitempty"`  // Reasoning data
}
```

### Main Methods
```go
func Quick(model, message string) (*Response, error)
func New(opts ...ClientOption) *Client
func (c *Client) Complete(ctx context.Context, req *Request) (*Response, error)
func (c *Client) Stream(ctx context.Context, req *Request) (StreamReader, error)
```

## License

Apache License

---

**LiteLLM** - Making LLM API calls simple and elegant