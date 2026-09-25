package wire

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestCheck(t *testing.T) {
	allowed := []string{"a", "b"}
	if err := CheckOptions(map[string]any{"a": 1, "b": 2}, allowed); err != nil {
		t.Fatal(err)
	}
	// The first unknown key in sorted order is reported, deterministically.
	err := CheckOptions(map[string]any{"z": 1, "c": 2, "a": 3}, allowed)
	if err == nil || err.Error() != `unsupported provider option "c"` {
		t.Fatalf("err = %v", err)
	}
}

func TestPut(t *testing.T) {
	for _, tc := range []struct {
		name    string
		current any
		value   any
		want    any
		wantErr bool
	}{
		{"new field", nil, "v", "v", false},
		{"object merges", map[string]any{"a": 1}, map[string]any{"c": 3}, map[string]any{"a": 1, "c": 3}, false},
		{"nested object merges", map[string]any{"o": map[string]any{"a": 1}}, map[string]any{"o": map[string]any{"b": 2}}, map[string]any{"o": map[string]any{"a": 1, "b": 2}}, false},
		{"nested scalar collision", map[string]any{"o": map[string]any{"a": 1}}, map[string]any{"o": map[string]any{"a": 2}}, nil, true},
		{"array appends", []any{1}, []any{2, 3}, []any{1, 2, 3}, false},
		{"scalar collision", "generated", "option", nil, true},
		{"object and scalar", map[string]any{"a": 1}, "option", nil, true},
		{"array and object", []any{1}, map[string]any{"a": 1}, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := map[string]any{}
			if tc.current != nil {
				body["k"] = tc.current
			}
			before, _ := json.Marshal(tc.current)
			err := PutOption(body, "k", tc.value)
			if tc.name == "nested scalar collision" && (err == nil || !strings.Contains(err.Error(), `"k.o.a"`)) {
				t.Fatalf("err = %v, want the collision path", err)
			}
			if tc.wantErr {
				if err == nil || !strings.Contains(err.Error(), "conflicts with a generated request field") {
					t.Fatalf("err = %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(body["k"], tc.want) {
				t.Fatalf("body[k] = %#v, want %#v", body["k"], tc.want)
			}
			// The generated value is replaced, not modified in place.
			if after, _ := json.Marshal(tc.current); string(after) != string(before) {
				t.Fatalf("generated value mutated: %s", after)
			}
		})
	}
}

func TestApplyStopsAtFirstConflictInKeyOrder(t *testing.T) {
	body := map[string]any{"b": 1, "c": 1}
	err := ApplyOptions(body, map[string]any{"c": 2, "a": 2, "b": 2})
	if err == nil || !strings.Contains(err.Error(), `"b"`) {
		t.Fatalf("err = %v", err)
	}
	if body["a"] != 2 || body["c"] != 1 {
		t.Fatalf("body = %#v", body)
	}
}

func TestMarshal(t *testing.T) {
	type wire struct {
		Model    string         `json:"model"`
		Seed     int64          `json:"seed"`
		Config   map[string]any `json:"config,omitempty"`
		Optional string         `json:"optional,omitempty"`
	}
	w := wire{Model: "m", Seed: 9007199254740993, Config: map[string]any{"a": 1}}
	plain, err := MarshalBody(w, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want, _ := json.Marshal(w); string(plain) != string(want) {
		t.Fatalf("without options = %s, want %s", plain, want)
	}
	data, err := MarshalBody(w, map[string]any{"config": map[string]any{"b": 2}, "extra": json.Number("12345678901234567890")})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"config":{"a":1,"b":2},"extra":12345678901234567890,"model":"m","seed":9007199254740993}`; string(data) != want {
		t.Fatalf("with options = %s, want %s", data, want)
	}
	if _, err := MarshalBody(w, map[string]any{"model": "other"}); err == nil {
		t.Fatal("option overriding a generated scalar was accepted")
	}
}

func TestMarshalKeepsGeneratedKeyOrder(t *testing.T) {
	body := struct {
		Schema json.RawMessage `json:"schema"`
		Config json.RawMessage `json:"config"`
	}{json.RawMessage(`{"z":1,"a":2}`), json.RawMessage(`{"s":{"z":1,"a":2}}`)}
	data, err := MarshalBody(body, map[string]any{"config": map[string]any{"b": true}})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"config":{"b":true,"s":{"z":1,"a":2}},"schema":{"z":1,"a":2}}`; string(data) != want {
		t.Fatalf("body = %s, want %s", data, want)
	}
}
