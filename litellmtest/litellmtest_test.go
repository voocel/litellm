package litellmtest_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/litellmtest"
)

func TestScriptedReplies(t *testing.T) {
	state := &litellm.ProviderState{Provider: "test", Data: json.RawMessage(`{"sig":"s"}`)}
	call := litellm.ToolUseBlock{ID: "c1", Name: "read", Arguments: `{"path":"a"}`, State: state}
	p := litellmtest.New(
		litellmtest.Reply{Blocks: []litellm.Block{litellm.ReasoningBlock{Text: "hm", State: state}, call}, Usage: litellm.Usage{InputTokens: 3}},
		litellmtest.Text("done"),
		litellmtest.Fail(litellm.NewError("test", litellm.ErrorTypeRateLimit, "slow down", nil)),
	)
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	stream, err := client.Stream(ctx, litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := litellm.Collect(stream)
	stream.Close()
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Block{litellm.ReasoningBlock{Text: "hm", State: state}, call}
	if !reflect.DeepEqual(resp.Blocks, want) || resp.FinishReason != litellm.FinishReasonToolCall || resp.Usage.InputTokens != 3 {
		t.Fatalf("first reply = %#v", resp)
	}

	resp, err = client.Chat(ctx, litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("again")}})
	if err != nil || resp.Text() != "done" || resp.FinishReason != litellm.FinishReasonStop {
		t.Fatalf("second reply = %#v, %v", resp, err)
	}

	if _, err := client.Stream(ctx, litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("x")}}); litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit {
		t.Fatalf("third reply err = %v", err)
	}
	if _, err := client.Chat(ctx, litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("x")}}); err == nil {
		t.Fatal("a call beyond the script succeeded")
	}

	reqs := p.Requests()
	if len(reqs) != 4 || reqs[1].Messages[0].Blocks[0].(litellm.TextBlock).Text != "again" {
		t.Fatalf("requests = %#v", reqs)
	}
}

func TestStreamStopsWhenCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := litellmtest.New(litellmtest.Text("hi")).Stream(ctx, &litellm.Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if _, err := stream.Next(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Next after cancel = %v", err)
	}
}

func TestStreamErrAndStall(t *testing.T) {
	boom := litellm.NewError("test", litellm.ErrorTypeOverloaded, "busy", nil)
	p := litellmtest.New(
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("par")}, StreamErr: boom},
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("wait")}, Stall: true},
	)
	client, _ := litellm.New(p)
	stream, err := client.Stream(context.Background(), litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := litellm.Collect(stream)
	stream.Close()
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeOverloaded || resp.Text() != "par" {
		t.Fatalf("failed stream = %q, %v", resp.Text(), err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	stream, err = client.Stream(ctx, litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var got string
	resp, err = litellm.Handle(stream, func(ev litellm.Event) error {
		if d, ok := ev.(litellm.TextDelta); ok {
			got += d.Text
			cancel()
		}
		return nil
	})
	if !errors.Is(err, context.Canceled) || got != "wait" || resp.Text() != "wait" {
		t.Fatalf("stalled stream = %q, %v", got, err)
	}
}
