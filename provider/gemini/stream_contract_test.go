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

// Signed parts retain their boundaries, including a stream's trailing empty
// signature part. Replay must preserve the structure the server returned.
func TestStreamSignaturesMatchCompleteResponse(t *testing.T) {
	for _, tc := range []struct {
		name, complete string
		chunks         []string
		want           []litellm.Block
	}{{
		name:     "text signature in an empty part",
		complete: `{"candidates":[{"content":{"parts":[{"text":"Hello"},{"text":"","thoughtSignature":"sig"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.Text("Hello"), litellm.TextBlock{State: signed("m", "sig")}},
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
			litellm.ReasoningBlock{Text: "a", State: signed("m", "s1")},
			litellm.ReasoningBlock{Text: "b"},
			litellm.ReasoningBlock{Text: "c", State: signed("m", "s2")},
			litellm.TextBlock{Text: "answer"},
		},
	}, {
		name:     "unsigned text around a signed part",
		complete: `{"candidates":[{"content":{"parts":[{"text":"a"},{"text":"b","thoughtSignature":"sig"},{"text":"c"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b","thoughtSignature":"sig"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"c"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.Text("a"), litellm.TextBlock{Text: "b", State: signed("m", "sig")}, litellm.Text("c")},
	}, {
		name:     "text after a trailing signature stays separate",
		complete: `{"candidates":[{"content":{"parts":[{"text":"ab"},{"text":"","thoughtSignature":"sig"},{"text":"c"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"c"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.Text("ab"), litellm.TextBlock{State: signed("m", "sig")}, litellm.Text("c")},
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
			for _, blocks := range [][]litellm.Block{want.Blocks, got.Blocks} {
				contents, _, err := convertMessages([]litellm.Message{litellm.Assistant(blocks...)})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(contents[0].Parts, complete.Candidates[0].Content.Parts) {
					t.Fatalf("replay changed signed part boundaries: %+v", contents[0].Parts)
				}
			}
		})
	}
}
