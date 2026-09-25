package anthropic

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
)

const replayableContent = `[
	{"type":"thinking","thinking":"need lookup","signature":"sig"},
	{"type":"redacted_thinking","data":"opaque"},
	{"type":"text","text":"calling"},
	{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]`

const citation = `{"type":"web_search_result_location","cited_text":"quoted","url":"https://x.test","title":"t","encrypted_index":"e"}`

func TestConvertResponse(t *testing.T) {
	for _, test := range []struct {
		name string
		raw  string
		want *litellm.Response
	}{
		{
			name: "blocks usage and warnings",
			raw: `{"model":"claude-x","stop_reason":"tool_use",
				"usage":{"input_tokens":5,"output_tokens":9,"cache_read_input_tokens":2,"cache_creation_input_tokens":3},
				"content":[
					{"type":"thinking","thinking":"need lookup","signature":"sig"},
					{"type":"redacted_thinking","data":"opaque"},
					{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}},
					{"type":"text","text":"calling"},
					{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]}`,
			want: &litellm.Response{
				Blocks: []litellm.Block{
					litellm.ReasoningBlock{Text: "need lookup", State: &litellm.ProviderState{
						Provider: "anthropic", Model: "requested", Data: json.RawMessage(`{"type":"thinking","signature":"sig"}`),
					}},
					litellm.ReasoningBlock{State: &litellm.ProviderState{
						Provider: "anthropic", Model: "requested", Data: json.RawMessage(`{"type":"redacted_thinking","data":"opaque"}`),
					}},
					litellm.TextBlock{Text: "calling"},
					litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
				},
				Usage:           litellm.Usage{InputTokens: new(10), OutputTokens: new(9), TotalTokens: new(19), CacheReadTokens: new(2), CacheWriteTokens: new(3)},
				Model:           "claude-x",
				Provider:        "anthropic",
				FinishReason:    litellm.FinishReasonToolCall,
				FinishReasonRaw: "tool_use",
				Warnings: []litellm.Warning{{
					Code: "anthropic.unsupported_block", Provider: "anthropic",
					Message: `dropped content block "server_tool_use", which litellm does not model`,
				}},
			},
		},
		{
			name: "citations become annotations",
			raw:  `{"model":"m","stop_reason":"end_turn","usage":{},"content":[{"type":"text","text":"x","citations":[` + citation + `]}]}`,
			want: &litellm.Response{
				Blocks: []litellm.Block{litellm.TextBlock{Text: "x", Annotations: []litellm.Annotation{
					{Type: "web_search_result_location", Text: "quoted", URL: "https://x.test", Extra: json.RawMessage(citation)},
				}}},
				Model: "m", Provider: "anthropic", FinishReason: litellm.FinishReasonStop, FinishReasonRaw: "end_turn",
			},
		},
		{
			name: "model falls back and unknown usage stays unknown",
			raw:  `{"content":[],"stop_reason":"end_turn","usage":{}}`,
			want: &litellm.Response{Model: "requested", Provider: "anthropic", FinishReason: litellm.FinishReasonStop, FinishReasonRaw: "end_turn"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var resp response
			if err := json.Unmarshal([]byte(test.raw), &resp); err != nil {
				t.Fatal(err)
			}
			if got := convertResponse(&resp, "requested"); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("got  %#v\nwant %#v", got, test.want)
			}
		})
	}
}

// Response blocks replayed as history reproduce the wire content they came from.
func TestResponseBlocksReplayAsHistory(t *testing.T) {
	var resp response
	if err := json.Unmarshal([]byte(`{"content":`+replayableContent+`}`), &resp); err != nil {
		t.Fatal(err)
	}
	data, err := buildRequest(&litellm.Request{
		Model:     "claude",
		MaxTokens: new(1024),
		Messages:  []litellm.Message{litellm.UserText("hi"), litellm.Assistant(convertResponse(&resp, "claude").Blocks...)},
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	var body struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	if got := body.Messages[1].Content; !jsonEqual(t, got, replayableContent) {
		t.Fatalf("replayed content = %s", got)
	}
}
