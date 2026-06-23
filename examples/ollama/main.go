package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/ollama"
)

func main() {
	client, err := ollama.NewClient(ollama.Config{})
	if err != nil {
		log.Fatal(err)
	}

	stream, err := client.Stream(context.Background(), litellm.Request{
		Model: "llama3.2",
		Messages: []litellm.Message{
			litellm.UserText("Write a haiku about Go programming."),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()

	for {
		event, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		switch e := event.(type) {
		case litellm.ContentDelta:
			fmt.Print(e.Text)
		case litellm.DoneEvent:
			fmt.Println()
			return
		}
	}
}
