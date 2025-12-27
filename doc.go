/*
Package litellm provides a predictable, explicit client for multiple LLM providers.

# Design Principles

  - Explicit configuration: no environment auto-discovery, no auto-routing
  - Single-provider client: each Client binds exactly one Provider
  - Predictable behavior: fail fast instead of guessing

# Quick Start

Create a client explicitly:

	client, err := litellm.NewWithProvider("openai", litellm.ProviderConfig{
	    APIKey: os.Getenv("OPENAI_API_KEY"),
	})
	if err != nil {
	    log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), &litellm.Request{
	    Model: "gpt-4o-mini",
	    Messages: []litellm.Message{
	        {Role: "user", Content: "Explain AI in one sentence."},
	    },
	})

# Streaming

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

# OpenAI Responses API

Use a dedicated request type for Responses API:

	resp, err := client.Responses(ctx, &litellm.OpenAIResponsesRequest{
	    Model: "o3-mini",
	    Messages: []litellm.Message{
	        {Role: "user", Content: "Solve 15*8 step by step."},
	    },
	    ReasoningEffort:  "medium",
	    ReasoningSummary: "auto",
	    MaxOutputTokens:  litellm.IntPtr(800),
	})
	_ = resp

# Custom Providers

Implement the Provider interface and register it:

	type MyProvider struct{}
	func (p *MyProvider) Name() string { return "myprovider" }
	func (p *MyProvider) Validate() error { return nil }
	func (p *MyProvider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) { ... }
	func (p *MyProvider) Stream(ctx context.Context, req *litellm.Request) (litellm.StreamReader, error) { ... }

	litellm.RegisterProvider("myprovider", func(cfg litellm.ProviderConfig) litellm.Provider {
	    return &MyProvider{}
	})

# Thread Safety

Client is safe for concurrent use. StreamReader is not goroutine-safe and must be consumed by a single goroutine.
*/
package litellm
