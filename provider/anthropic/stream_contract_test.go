package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestStreamContentMatchesCompleteResponse(t *testing.T) {
	var complete anthropicResponse
	if err := json.Unmarshal([]byte(`{"model":"claude","stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3},"content":[{"type":"text","text":"hello"},{"type":"text","text":""},{"type":"thinking","thinking":"think","signature":"sig"},{"type":"redacted_thinking","data":"opaque"}]}`), &complete); err != nil {
		t.Fatal(err)
	}
	want, err := convertResponse(&complete, "claude")
	if err != nil {
		t.Fatal(err)
	}
	stream := newStream(streamResponse(strings.Join([]string{
		`data: {"type":"message_start","message":{"model":"claude","usage":{"input_tokens":2}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hel"}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`data: {"type":"content_block_stop","index":1}`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"think"}}`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sig"}}`,
		`data: {"type":"content_block_stop","index":2}`,
		`data: {"type":"content_block_start","index":3,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		`data: {"type":"content_block_stop","index":3}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")), &litellm.Request{Model: "claude"}, nil)
	defer stream.Close()
	got, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream=%#v\ncomplete=%#v\nstream blocks=%#v\ncomplete blocks=%#v", got, want, got.Blocks, want.Blocks)
	}
}
