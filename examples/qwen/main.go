package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/qwen"
)

func main() {
	client, err := qwen.NewClient(qwen.Config{
		APIKey:  os.Getenv("QWEN_API_KEY"),
		BaseURL: os.Getenv("QWEN_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "qwen-plus",
		Messages: []litellm.Message{
			litellm.UserText("Explain DashScope's OpenAI-compatible endpoint in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
