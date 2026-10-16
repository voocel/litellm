package wire

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
)

func TestState(t *testing.T) {
	type sig struct {
		Signature string `json:"signature"`
	}
	state := NewState("p", "m", sig{"s"})
	if state.Provider != "p" || state.Model != "m" || string(state.Data) != `{"signature":"s"}` {
		t.Fatalf("NewState = %+v", state)
	}
	if raw := NewState("p", "m", json.RawMessage(`{ "kept" : 1 }`)); string(raw.Data) != `{ "kept" : 1 }` {
		t.Fatalf("raw data = %s", raw.Data)
	}
	if got, ok := ReadState[sig](state, "p"); !ok || got.Signature != "s" {
		t.Fatalf("own state = %+v, %v", got, ok)
	}
	for name, state := range map[string]*litellm.ProviderState{
		"absent":      nil,
		"foreign":     state,
		"undecodable": {Provider: "q", Data: json.RawMessage(`[1]`)},
	} {
		if _, ok := ReadState[sig](state, "q"); ok {
			t.Errorf("%s state was read", name)
		}
	}
}

func TestWithoutAssistantImages(t *testing.T) {
	img := litellm.ImageBlock{Data: []byte("png"), MIME: "image/png"}
	assistant := litellm.Assistant(litellm.Text("a"), img, litellm.Text("b"))
	got := WithoutAssistantImages(assistant)
	if want := []litellm.Block{litellm.Text("a"), litellm.Text("b")}; !reflect.DeepEqual(got, want) {
		t.Fatalf("blocks %#v", got)
	}
	if _, ok := assistant.Blocks[1].(litellm.ImageBlock); !ok {
		t.Error("the message's own blocks were changed")
	}
	if user := litellm.User(img); len(WithoutAssistantImages(user)) != 1 {
		t.Error("a user's image was dropped")
	}
}
