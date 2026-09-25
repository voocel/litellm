package wire

import (
	"encoding/json"

	"github.com/voocel/litellm"
)

// NewState returns data, the vendor's native replay fields, as the state of a
// block provider produced for model. json.RawMessage data is kept verbatim.
func NewState(provider, model string, data any) *litellm.ProviderState {
	raw, ok := data.(json.RawMessage)
	if !ok {
		raw, _ = json.Marshal(data) // replay fields are plain JSON values
	}
	return &litellm.ProviderState{Provider: provider, Model: model, Data: raw}
}

// ReadState decodes state into a T when provider produced it. Foreign, absent
// or undecodable state reports false.
func ReadState[T any](state *litellm.ProviderState, provider string) (T, bool) {
	var v T
	if state == nil || state.Provider != provider {
		return v, false
	}
	return v, json.Unmarshal(state.Data, &v) == nil
}
