/*
Package litellm provides a small, explicit multi-provider LLM SDK core.

The root package owns the provider-agnostic domain model: Request, Response,
Message, Block, Stream, Event, structured errors, warnings, and observers. Pricing lives in its optional subpackage. Concrete providers live in provider-specific subpackages.

# Quick Start

Create a provider with its package-specific config, then bind a Client:

	import (
	    "context"
	    "fmt"
	    "os"

	    "github.com/voocel/litellm"
	    "github.com/voocel/litellm/provider/anthropic"
	)

	provider, err := anthropic.New(anthropic.Config{
	    APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	})
	if err != nil {
	    panic(err)
	}
	client, err := litellm.New(provider)
	if err != nil {
	    panic(err)
	}

	maxTokens := 1024
	resp, err := client.Chat(context.Background(), litellm.Request{
	    Model:     "claude-sonnet-4-5",
	    MaxTokens: &maxTokens,
	    Messages:  []litellm.Message{litellm.UserText("Explain AI in one sentence.")},
	})
	if err != nil {
	    panic(err)
	}
	fmt.Println(resp.Text())

# Blocks

Message and Response content is represented as ordered Blocks. This preserves
the order of text, reasoning, tool use, tool results, cache markers, and opaque
provider signatures across multi-turn agent workflows.

# Streaming

Providers stream typed Event values. Client streams retain validation state by
default; complete content is retained only by explicit aggregation or observer
content capture. Use a type switch for real-time handling
or Collect to aggregate a stream into a Response:
Stream is intended for single-goroutine consumption; do not call Next
concurrently.

	stream, err := client.Stream(ctx, req)
	if err != nil {
	    panic(err)
	}
	defer stream.Close()

	for {
	    event, err := stream.Next()
	    if err != nil {
	        panic(err)
	    }
	    switch e := event.(type) {
	    case litellm.ContentStart:
	        if text, ok := e.Block.(litellm.TextBlock); ok {
	            fmt.Print(text.Text)
	        }
	    case litellm.ContentDelta:
	        fmt.Print(e.Text)
	    case litellm.DoneEvent:
	        return
	    }
	}

# Design

The SDK is intentionally not a gateway, router, agent runtime, account system,
or request scheduler. It binds one Client to one Provider and exposes explicit
configuration and structural validation. Message-history validation and repair
are explicit utilities; the Client never rewrites conversation history. Provider
options are JSON data, and optional usage counters distinguish unknown from zero.
*/
package litellm
