/*
Package litellm provides a unified interface for accessing multiple Large Language Model (LLM) platforms.

# Design Philosophy

LiteLLM is designed with simplicity and elegance in mind:

  - Single Entry Point: Only litellm.New() - no confusing choice between multiple APIs
  - Auto-Resolution: Models automatically resolve to correct providers (gpt-4o → OpenAI, claude → Anthropic)
  - Type-Safe Configuration: WithOpenAI(), WithAnthropic() instead of error-prone string-based config
  - Zero Configuration: Works immediately with environment variables
  - Provider-Agnostic: Same code works across all AI providers

# Quick Start

The simplest way to get started is using Quick():

	response, err := litellm.Quick("gemini-3.0-pro", "Hello, LiteLLM!")
	if err != nil {
	    log.Fatal(err)
	}
	fmt.Println(response.Content)

# Full Example

For production use, create a client with explicit configuration:

	client, err := litellm.New(
	    litellm.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
	    litellm.WithAnthropic(os.Getenv("ANTHROPIC_API_KEY")),
	    litellm.WithDefaults(2048, 0.7),
	    litellm.WithRetries(3, 1*time.Second),
	)
	if err != nil {
	    log.Fatal(err)
	}

	response, err := client.Chat(context.Background(), &litellm.Request{
	    Model: "gpt-5.1-mini",
	    Messages: []litellm.Message{
	        {Role: "user", Content: "Explain quantum computing"},
	    },
	    MaxTokens:   litellm.IntPtr(500),
	    Temperature: litellm.Float64Ptr(0.8),
	})

# Streaming

Real-time streaming responses:

	stream, err := client.Stream(ctx, &litellm.Request{
	    Model: "claude-4.5-sonnet",
	    Messages: []litellm.Message{
	        {Role: "user", Content: "Write a story"},
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

# Function Calling

Unified tool calling across providers:

	tools := []litellm.Tool{
	    {
	        Type: "function",
	        Function: litellm.FunctionDef{
	            Name:        "get_weather",
	            Description: "Get weather information",
	            Parameters:  schema,
	        },
	    },
	}

	response, err := client.Chat(ctx, &litellm.Request{
	    Model:      "gpt-5.1",
	    Messages:   messages,
	    Tools:      tools,
	    ToolChoice: "auto",
	})

# Structured Outputs

JSON Schema validation for reliable responses:

	response, err := client.Chat(ctx, &litellm.Request{
	    Model:    "gpt-4o",
	    Messages: messages,
	    ResponseFormat: litellm.NewResponseFormatJSONSchema(
	        "person",
	        "A person's profile",
	        schema,
	        true, // strict mode
	    ),
	})

# Reasoning Models

Support for OpenAI o-series and other reasoning models:

	response, err := client.Chat(ctx, &litellm.Request{
	    Model:            "gpt-5.1",
	    Messages:         messages,
	    ReasoningEffort:  "medium",
	    ReasoningSummary: "detailed",
	    MaxTokens:        litellm.IntPtr(1000),
	})

	if response.Reasoning != nil {
	    fmt.Printf("Reasoning: %s\n", response.Reasoning.Summary)
	}

# Supported Providers

  - OpenAI: GPT-5, GPT-4o, o-series reasoning models
  - Anthropic: Claude 4/4.5 family
  - Google Gemini: Gemini 2.5/3.0 Pro/Flash
  - DeepSeek: Chat and Reasoner models
  - Qwen: Alibaba's Qwen3-Coder family
  - GLM: ZhiPu AI's GLM-4.6 family
  - OpenRouter: 200+ models from multiple providers

# Package Layout

The `providers` subpackage contains builtin provider implementations and is
considered an internal detail. End users should only import `litellm`.
If you need a custom provider, implement `litellm.Provider` and register it
with `litellm.RegisterProvider`.

# Custom Providers

Extend with your own providers:

	type MyProvider struct {
	    // implement litellm.Provider interface
	}

	litellm.RegisterProvider("myprovider", NewMyProvider)

	client, err := litellm.New(
	    litellm.WithProviderConfig("myprovider", config),
	)

# Error Handling

Structured error types with retry information:

	response, err := client.Chat(ctx, req)
	if err != nil {
	    if litellm.IsRateLimitError(err) {
	        retryAfter := litellm.GetRetryAfter(err)
	        log.Printf("Rate limited, retry after %d seconds", retryAfter)
	    } else if litellm.IsRetryableError(err) {
	        log.Printf("Retryable error: %v", err)
	    } else {
	        log.Printf("Permanent error: %v", err)
	    }
	}

# Environment Variables

Auto-discovery uses these environment variables:

	OPENAI_API_KEY       - OpenAI API key
	ANTHROPIC_API_KEY    - Anthropic API key
	GEMINI_API_KEY       - Google Gemini API key
	DEEPSEEK_API_KEY     - DeepSeek API key
	QWEN_API_KEY         - Alibaba Qwen API key
	GLM_API_KEY          - ZhiPu GLM API key
	OPENROUTER_API_KEY   - OpenRouter API key

# Resilience Configuration

Optional retry mechanism with exponential backoff:

	client, err := litellm.New(
	    litellm.WithOpenAI(apiKey),
	    litellm.WithRetries(3, 1*time.Second),
	    litellm.WithTimeout(60*time.Second),
	)

# Thread Safety

The Client is safe for concurrent use. However, StreamReader instances
are NOT thread-safe and should be used by a single goroutine at a time.
Always call defer stream.Close() to prevent resource leaks.

# Best Practices

1. Reuse Client instances - they maintain connection pools
2. Always defer stream.Close() when using streaming
3. Use context for cancellation and timeouts
4. Handle errors with type-specific checks
5. Use WithRetries() for production resilience

For more examples, see https://github.com/voocel/litellm/tree/main/examples
*/
package litellm
