package openai

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestResponsesContentSnapshotMatchesCompleteResponse(t *testing.T) {
	const finalPart = `{"type":"output_text","text":"hello","annotations":[{"type":"url_citation","url":"https://example.com","title":"source"}],"logprobs":[{"token":"hello","logprob":-0.1}]}`
	var complete responsesResponse
	if err := json.Unmarshal([]byte(`{"model":"m","status":"completed","output":[{"type":"message","content":[`+finalPart+`,{"type":"output_text","text":""}]}]}`), &complete); err != nil {
		t.Fatal(err)
	}
	want, err := convertResponsesResponse(&complete, "m")
	if err != nil {
		t.Fatal(err)
	}
	stream := newResponsesStream(streamResponse(strings.Join([]string{
		`data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""},"sequence_number":1}`,
		`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hello","sequence_number":2}`,
		`data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":` + finalPart + `,"sequence_number":3}`,
		`data: {"type":"response.content_part.added","output_index":0,"content_index":1,"part":{"type":"output_text","text":""},"sequence_number":4}`,
		`data: {"type":"response.content_part.done","output_index":0,"content_index":1,"part":{"type":"output_text","text":""},"sequence_number":5}`,
		`data: {"type":"response.completed","response":{"model":"m","status":"completed","usage":{}},"sequence_number":6}`,
		``,
	}, "\n")), "m")
	defer stream.Close()
	got, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream=%#v\ncomplete=%#v\nstream blocks=%#v\ncomplete blocks=%#v", got, want, got.Blocks, want.Blocks)
	}
}
