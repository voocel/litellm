package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/examples/internal/exampleutil"
	"github.com/voocel/litellm/provider/gemini"
)

func main() {
	mode := "stream"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}

	provider, err := gemini.New(gemini.Config{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		BaseURL: os.Getenv("GEMINI_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}
	client, err := litellm.New(provider)
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	switch mode {
	case "chat":
		runChat(ctx, client)
	case "stream":
		runStream(ctx, client)
	default:
		log.Fatalf("unknown mode %q; use one of: chat | stream", mode)
	}
}

func model() string {
	if m := os.Getenv("GEMINI_MODEL"); m != "" {
		return m
	}
	return "gemini-3.8-flash"
}

func runChat(ctx context.Context, client *litellm.Client) {
	resp, err := client.Chat(ctx, litellm.Request{
		Model: model(),
		Messages: []litellm.Message{
			litellm.UserText("Explain multimodal prompts in one sentence."),
		},
		MaxTokens: new(1024),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}

func runStream(ctx context.Context, client *litellm.Client) {
	resp, err := exampleutil.Stream(ctx, client, litellm.Request{
		Model: model(),
		Messages: []litellm.Message{
			litellm.UserText("Explain multimodal prompts in one sentence."),
		},
		MaxTokens: new(1024),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
	fmt.Println()
	exampleutil.PrintUsage(resp.Usage)
}
