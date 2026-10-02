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
			litellm.ReasoningBlock{Text: "think", State: testState(`{"signature":"sig"}`)},
			litellm.ReasoningBlock{State: testState(`{"redactedContent":"b3BhcXVl"}`)},
			litellm.TextBlock{Text: "hello"},
			litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup", Arguments: `{"q":"x"}`},
		},
		// Input counts cache reads and writes.
		Usage:           litellm.Usage{InputTokens: 10, OutputTokens: 7, CacheReadTokens: 2, CacheWriteTokens: 3},
		Model:           "m",
		Provider:        "bedrock",
		FinishReason:    litellm.FinishReasonToolCall,
		FinishReasonRaw: "tool_use",
	}
	if got := convertResponse(&resp, "bedrock", "m"); !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %#v\nwant %#v", got, want)
	}
}

func testState(data string) *litellm.ProviderState {
	return &litellm.ProviderState{Provider: "bedrock", Model: "m", Data: json.RawMessage(data)}
}
