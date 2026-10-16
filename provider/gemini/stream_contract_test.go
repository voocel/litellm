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
	want, err := convertResponse(&complete, "gemini", "m")
	if err != nil {
		t.Fatal(err)
	}
	stream := newStream(streamResponse(testgolden.ReadFixtureString(t, "../../testdata/gemini/stream.jsonl")), "gemini", "m")
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
		want: []litellm.Block{litellm.Text("Hello"), litellm.TextBlock{State: signed("gemini", "m", "sig")}},
	}, {
		name:     "signed text parts stay apart",
		complete: `{"candidates":[{"content":{"parts":[{"text":"a","thoughtSignature":"s1"},{"text":"b","thoughtSignature":"s2"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a","thoughtSignature":"s1"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b","thoughtSignature":"s2"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.TextBlock{Text: "a", State: signed("gemini", "m", "s1")}, litellm.TextBlock{Text: "b", State: signed("gemini", "m", "s2")}},
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
			litellm.ReasoningBlock{Text: "a", State: signed("gemini", "m", "s1")},
			litellm.ReasoningBlock{Text: "b"},
			litellm.ReasoningBlock{Text: "c", State: signed("gemini", "m", "s2")},
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
		want: []litellm.Block{litellm.Text("a"), litellm.TextBlock{Text: "b", State: signed("gemini", "m", "sig")}, litellm.Text("c")},
	}, {
		name:     "text after a trailing signature stays separate",
		complete: `{"candidates":[{"content":{"parts":[{"text":"ab"},{"text":"","thoughtSignature":"sig"},{"text":"c"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"a"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"b"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"sig"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"c"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{litellm.Text("ab"), litellm.TextBlock{State: signed("gemini", "m", "sig")}, litellm.Text("c")},
	}, {
		name: "generated images keep their signatures",
		complete: `{"candidates":[{"content":{"parts":[
			{"text":"Here","thoughtSignature":"s1"},
			{"inlineData":{"mimeType":"image/png","data":"cG5n"},"thoughtSignature":"s2"},
			{"text":"and"},
			{"inlineData":{"mimeType":"image/jpeg","data":"anBn"},"thoughtSignature":"s3"}]},"finishReason":"STOP"}]}`,
		chunks: []string{
			`{"candidates":[{"content":{"parts":[{"text":"Here","thoughtSignature":"s1"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"cG5n"},"thoughtSignature":"s2"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"text":"and"}]}}]}`,
			`{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/jpeg","data":"anBn"},"thoughtSignature":"s3"}]},"finishReason":"STOP"}]}`,
		},
		want: []litellm.Block{
			litellm.TextBlock{Text: "Here", State: signed("gemini", "m", "s1")},
			litellm.ImageBlock{Data: []byte("png"), MIME: "image/png", State: signed("gemini", "m", "s2")},
			litellm.Text("and"),
			litellm.ImageBlock{Data: []byte("jpg"), MIME: "image/jpeg", State: signed("gemini", "m", "s3")},
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			var complete response
			if err := json.Unmarshal([]byte(tc.complete), &complete); err != nil {
				t.Fatal(err)
			}
			want, err := convertResponse(&complete, "gemini", "m")
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(want.Blocks, tc.want) {
				t.Fatalf("complete blocks = %#v", want.Blocks)
			}
			stream := newStream(streamResponse("data: "+strings.Join(tc.chunks, "\ndata: ")+"\n"), "gemini", "m")
			defer stream.Close()
			got, err := litellm.Collect(stream)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("stream=%#v\ncomplete=%#v", got, want)
			}
			for _, blocks := range [][]litellm.Block{want.Blocks, got.Blocks} {
				contents, _, err := convertMessages([]litellm.Message{litellm.Assistant(blocks...)}, "gemini")
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

// Thought images, unsigned drafts of the final one, have no block, nor has
// inline data other than an image, which is reported instead.
func TestInlineDataWithoutBlock(t *testing.T) {
	parts := []string{
		`{"text":"plan","thought":true}`,
		`{"inlineData":{"mimeType":"image/png","data":"ZHJhZnQ="},"thought":true}`,
		`{"text":" more","thought":true}`,
		`{"inlineData":{"mimeType":"image/png","data":"cG5n"},"thoughtSignature":"s"}`,
		`{"inlineData":{"mimeType":"audio/L16","data":"cGNt"}}`,
	}
	var complete response
	if err := json.Unmarshal([]byte(`{"candidates":[{"content":{"parts":[`+strings.Join(parts, ",")+`]},"finishReason":"STOP"}]}`), &complete); err != nil {
		t.Fatal(err)
	}
	want, err := convertResponse(&complete, "gemini", "m")
	if err != nil {
		t.Fatal(err)
	}
	blocks := []litellm.Block{
		litellm.ReasoningBlock{Text: "plan more"},
		litellm.ImageBlock{Data: []byte("png"), MIME: "image/png", State: signed("gemini", "m", "s")},
	}
	if !reflect.DeepEqual(want.Blocks, blocks) {
		t.Fatalf("complete blocks = %#v", want.Blocks)
	}
	warnings := []litellm.Warning{{Code: "gemini.unsupported_part", Provider: "gemini", Message: `dropped inline data of type "audio/L16", which litellm does not model`}}
	if !reflect.DeepEqual(want.Warnings, warnings) {
		t.Fatalf("complete warnings = %#v", want.Warnings)
	}
	var chunks []string
	for _, p := range parts {
		chunks = append(chunks, `data: {"candidates":[{"content":{"parts":[`+p+`]}}]}`)
	}
	chunks = append(chunks, `data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}`)
	got, err := litellm.Collect(newStream(streamResponse(strings.Join(chunks, "\n")), "gemini", "m"))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("stream=%#v, %v\ncomplete=%#v", got, err, want)
	}
}

// An empty unsigned thought part has no block, and a STOP needs no warning
// for its message, as an Antigravity proxy sends both.
func TestEmptyThoughtAndStopMessage(t *testing.T) {
	parts := []string{
		`{"thought":true,"text":""}`,
		`{"functionCall":{"id":"c","name":"f","args":{}},"thoughtSignature":"s"}`,
	}
	const finish = `"finishReason":"STOP","finishMessage":"Model generated function call(s)."`
	var complete response
	if err := json.Unmarshal([]byte(`{"candidates":[{"content":{"parts":[`+strings.Join(parts, ",")+`]},`+finish+`}]}`), &complete); err != nil {
		t.Fatal(err)
	}
	want, err := convertResponse(&complete, "gemini", "m")
	if err != nil {
		t.Fatal(err)
	}
	blocks := []litellm.Block{litellm.ToolUseBlock{ID: "c", Name: "f", Arguments: "{}", State: signed("gemini", "m", "s")}}
	if !reflect.DeepEqual(want.Blocks, blocks) || want.Warnings != nil {
		t.Fatalf("complete blocks = %#v, warnings = %#v", want.Blocks, want.Warnings)
	}
	var chunks []string
	for _, p := range parts {
		chunks = append(chunks, `data: {"candidates":[{"content":{"parts":[`+p+`]}}]}`)
	}
	chunks = append(chunks, `data: {"candidates":[{"content":{"parts":[]},`+finish+`}]}`)
	got, err := litellm.Collect(newStream(streamResponse(strings.Join(chunks, "\n")), "gemini", "m"))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("stream=%#v, %v\ncomplete=%#v", got, err, want)
	}
}
