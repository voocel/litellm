package wire

import (
	"encoding/json"
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
