package exampleutil

import (
	"context"
	"fmt"

	"github.com/voocel/litellm"
)

type StreamPrinter struct {
	reasoning bool
	answer    bool
}

func (p *StreamPrinter) WriteReasoning(text string) {
	if !p.reasoning {
		fmt.Println("reasoning:")
		p.reasoning = true
	}
	fmt.Print(text)
}

func (p *StreamPrinter) WriteAnswer(text string) {
	if !p.answer {
		if p.reasoning {
			fmt.Println()
			fmt.Println()
		}
		fmt.Println("answer:")
		p.answer = true
	}
	fmt.Print(text)
}

// Print is a litellm.Handle callback that prints reasoning and answer text.
func (p *StreamPrinter) Print(event litellm.Event) error {
	switch e := event.(type) {
	case litellm.ReasoningDelta:
		p.WriteReasoning(e.Text)
	case litellm.TextDelta:
		p.WriteAnswer(e.Text)
	}
	return nil
}

// Stream runs req and prints reasoning and answer text as they arrive.
func Stream(ctx context.Context, client *litellm.Client, req litellm.Request) (*litellm.Response, error) {
	stream, err := client.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	var printer StreamPrinter
	return litellm.Handle(stream, printer.Print)
}

func PrintUsage(usage litellm.Usage) {
	if !usage.HasTokens() {
		return
	}
	fmt.Printf("usage: input=%s output=%s total=%s reasoning=%s cache_read=%s cache_write=%s\n",
		tokenCount(usage.Input()),
		tokenCount(usage.Output()),
		tokenCount(usage.Total()),
		tokenCount(usage.Reasoning()),
		tokenCount(usage.CacheRead()),
		tokenCount(usage.CacheWrite()),
	)
}

func tokenCount(count int, known bool) string {
	if !known {
		return "unknown"
	}
	return fmt.Sprint(count)
}
