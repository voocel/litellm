package openai

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

// A stream aggregates to the response its response.completed event carries,
// including metadata that only content_part.done and output_item.done deliver.
func TestResponsesStreamMatchesCompleteResponse(t *testing.T) {
	fixture := testgolden.ReadFixtureString(t, "../../testdata/openai/responses_stream.sse")
	var completed struct {
		Response responsesResponse `json:"response"`
	}
	last := fixture[strings.LastIndex(fixture, "data: ")+len("data: "):]
	if err := json.Unmarshal([]byte(last), &completed); err != nil {
		t.Fatal(err)
	}
	want := convertResponsesResponse(&completed.Response, "openai", "m")

	p, got := testProvider(t, Config{API: APIResponses}, fixture)
	stream, err := p.Stream(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var ended []litellm.Block
	resp, err := litellm.Handle(stream, func(event litellm.Event) error {
		if end, ok := event.(litellm.BlockEnd); ok {
			ended = append(ended, end.Block)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("stream response:\n%#v\ncomplete response:\n%#v", resp, want)
	}
	if !reflect.DeepEqual(ended, want.Blocks) {
		t.Fatalf("BlockEnd blocks:\n%#v\nwant:\n%#v", ended, want.Blocks)
	}
	if got.req.Header.Get("Accept") != "text/event-stream" {
		t.Fatalf("Accept = %q", got.req.Header.Get("Accept"))
	}
	assertJSON(t, got.body, `{"model":"m","stream":true,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
}
