/*
Package litellm provides a small, explicit multi-provider LLM SDK core.

The root package owns the provider-agnostic domain model: Request, Response,
Message, Block, Stream, Event, structured errors, warnings, and observers.
Concrete providers live in provider subpackages, and the providers package
builds them by name; compat connects any OpenAI-compatible endpoint. catalog
and retry are optional utilities.

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
the order of text, reasoning, tool use, and tool results across multi-turn
agent workflows. Data a vendor needs back, such as a reasoning signature,
travels in a block's ProviderState and is sent only to the provider that
produced it.

# Streaming

A stream is a sequence of blocks. BlockStart opens the block at Index, the
position it takes in Response.Blocks; deltas of the same kind grow it; BlockEnd
closes it with the completed block. Blocks may interleave, and all end before
DoneEvent. Handle aggregates the stream and passes each event to a callback;
Collect only aggregates. Stream is for single-goroutine consumption.

	stream, err := client.Stream(ctx, req)
	if err != nil {
	    panic(err)
	}
	defer stream.Close()

	resp, err := litellm.Handle(stream, func(event litellm.Event) error {
	    if e, ok := event.(litellm.TextDelta); ok {
	        fmt.Print(e.Text)
	    }
	    return nil
	})

# Design

The SDK is intentionally not a gateway, router, agent runtime, account system,
or request scheduler. It binds one Client to one Provider and exposes explicit
configuration and structural validation. It maps structure only: it does not
infer model features, validate vendor values locally, or rewrite user input;
the vendor API is the authority. ProviderOptions carry native wire fields, and
optional usage counters distinguish unknown from zero.
*/
package litellm
