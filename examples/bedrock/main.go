package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/bedrock"
)

func main() {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = "us-east-1"
	}
	client, err := bedrock.NewClient(bedrock.Config{
		Region: region,
		Credentials: bedrock.StaticCredentials(
			os.Getenv("AWS_ACCESS_KEY_ID"),
			os.Getenv("AWS_SECRET_ACCESS_KEY"),
			os.Getenv("AWS_SESSION_TOKEN"),
		),
	})
	if err != nil {
		log.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), litellm.Request{
		Model: "anthropic.claude-3-5-sonnet-20240620-v1:0",
		Messages: []litellm.Message{
			litellm.UserText("Explain Amazon Bedrock in one sentence."),
		},
		MaxTokens: litellm.IntPtr(256),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Text())
}
