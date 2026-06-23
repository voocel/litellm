package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/minimax"
)

func main() {
	client, err := minimax.NewClient(minimax.Config{
		APIKey:  os.Getenv("MINIMAX_API_KEY"),
		BaseURL: os.Getenv("MINIMAX_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "MiniMax-M3",
		Messages: []litellm.Message{
			litellm.UserText("Explain MiniMax in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
		Thinking:  &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
	if reasoning := resp.Reasoning(); reasoning != "" {
		fmt.Println("\nreasoning:", reasoning)
	}
}
