package gemini

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestStreamCollectMatchesCompleteResponse(t *testing.T) {
	var complete response
	if err := json.Unmarshal([]byte(`{
		"candidates":[{"content":{"role":"model","parts":[
			{"text":"think","thought":true,"thoughtSignature":"sig-think"},
			{"text":"answer"},
			{"thoughtSignature":"sig-call","functionCall":{"id":"call_1","name":"lookup","args":{"q":"x"}}}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":9,"cachedContentTokenCount":1}
	}`), &complete); err != nil {
		t.Fatal(err)
	}
	want := convertResponse(&complete, "m")
	stream := newStream(streamResponse(testgolden.ReadFixtureString(t, "../../testdata/gemini/stream.jsonl")), "m")
	defer stream.Close()
	got, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream=%#v\ncomplete=%#v", got, want)
	}
}

// Signatures attach to their run, but two signed parts never merge; streams
// may send a text run's signature in a trailing empty part.
func TestStreamSignaturesMatchCompleteResponse(t *testing.T) {
	for _, tc := range []struct {
		name, complete string
		chunks         []string
		want           []litellm.Block
	}{{
		name:     "text signature in an empty part",
		complete: `{"candidates":[{"content":{"parts":[{"text":"Hello","thoughtSignature":"sig"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.TextBlock{Text: "Hello", State: signed("m", "sig")}},
	}, {
		name:     "signed text parts stay apart",
		complete: `{"candidates":[{"content":{"parts":[{"text":"a","thoughtSignature":"s1"},{"text":"b","thoughtSignature":"s2"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a","thoughtSignature":"s1"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b","thoughtSignature":"s2"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.TextBlock{Text: "a", State: signed("m", "s1")}, litellm.TextBlock{Text: "b", State: signed("m", "s2")}},
	}, {
		name: "signed thought parts stay apart",
		complete: `{"candidates":[{"content":{"parts":[
			{"text":"a","thought":true,"thoughtSignature":"s1"},{"text":"b","thought":true},
			{"text":"c","thought":true,"thoughtSignature":"s2"},{"text":"answer"}]},
			"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"Malformed function call: f("}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a","thought":true,"thoughtSignature":"s1"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b","thought":true}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"c","thought":true,"thoughtSignature":"s2"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"answer"}]},"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"Malformed function call: f("}]}`,
		},
		want: []litellm.Block{
			litellm.ReasoningBlock{Text: "ab", State: signed("m", "s1")},
			litellm.ReasoningBlock{Text: "c", State: signed("m", "s2")},
			litellm.TextBlock{Text: "answer"},
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			var complete response
			if err := json.Unmarshal([]byte(tc.complete), &complete); err != nil {
				t.Fatal(err)
			}
			want := convertResponse(&complete, "m")
			if !reflect.DeepEqual(want.Blocks, tc.want) {
				t.Fatalf("complete blocks = %#v", want.Blocks)
			}
			stream := newStream(streamResponse("data: "+strings.Join(tc.chunks, "\ndata: ")+"\n"), "m")
			defer stream.Close()
			got, err := litellm.Collect(stream)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("stream=%#v\ncomplete=%#v", got, want)
			}
		})
	}
}
