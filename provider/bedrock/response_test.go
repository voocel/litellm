package bedrock

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
)

func TestConvertResponse(t *testing.T) {
	var resp response
	if err := json.Unmarshal([]byte(`{
		"output":{"message":{"role":"assistant","content":[
			{"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"}}},
			{"reasoningContent":{"redactedContent":"b3BhcXVl"}},
			{"text":"hello"},
			{"toolUse":{"toolUseId":"toolu_1","name":"lookup","input":{"q":"x"}}}]}},
		"stopReason":"tool_use",
		"usage":{"inputTokens":5,"outputTokens":7,"totalTokens":12,"cacheReadInputTokens":2,"cacheWriteInputTokens":3}
	}`), &resp); err != nil {
		t.Fatal(err)
	}
	want := &litellm.Response{
		Blocks: []litellm.Block{
			litellm.ReasoningBlock{Text: "think", Signature: "sig"},
			litellm.ReasoningBlock{Redacted: []byte("opaque")},
			litellm.TextBlock{Text: "hello"},
			litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
		},
		// Input counts cache reads and writes.
		Usage:           litellm.Usage{InputTokens: new(10), OutputTokens: new(7), TotalTokens: new(17), CacheReadTokens: new(2), CacheWriteTokens: new(3)},
		Model:           "m",
		Provider:        "bedrock",
		FinishReason:    litellm.FinishReasonToolCall,
		FinishReasonRaw: "tool_use",
	}
	if got := convertResponse(&resp, "m"); !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %#v\nwant %#v", got, want)
	}
}
