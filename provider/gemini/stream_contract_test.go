package gemini

import (
	"encoding/json"
	"reflect"
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
