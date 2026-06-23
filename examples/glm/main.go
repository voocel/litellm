package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/glm"
)

func main() {
	client, err := glm.NewClient(glm.Config{
		APIKey:  os.Getenv("GLM_API_KEY"),
		BaseURL: os.Getenv("GLM_BASE_URL"),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "glm-5.2",
		Messages: []litellm.Message{
			litellm.UserText("Explain Zhipu GLM in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
		Thinking:  &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
