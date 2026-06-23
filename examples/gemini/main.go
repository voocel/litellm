package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/gemini"
)

func main() {
	client, err := gemini.NewClient(gemini.Config{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		BaseURL: os.Getenv("GEMINI_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "gemini-2.5-flash",
		Messages: []litellm.Message{
			litellm.UserText("Explain multimodal prompts in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
