package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/grok"
)

func main() {
	client, err := grok.NewClient(grok.Config{
		APIKey:  os.Getenv("XAI_API_KEY"),
		BaseURL: os.Getenv("XAI_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "grok-4",
		Messages: []litellm.Message{
			litellm.UserText("Explain Grok in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
		Thinking:  &litellm.Thinking{Mode: litellm.ThinkingEnabled, Level: "high"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
