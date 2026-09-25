package wire

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// CheckOptions rejects keys outside allowed, in sorted order.
func CheckOptions(options map[string]any, allowed []string) error {
	for _, key := range slices.Sorted(maps.Keys(options)) {
		if !slices.Contains(allowed, key) {
			return fmt.Errorf("unsupported provider option %q", key)
		}
	}
	return nil
}

// PutOption copies an option into body. An option naming a generated object is
// merged into it and one naming a generated array is appended to it, at every
// depth; any other collision is an error. Generated values are not modified.
func PutOption(body map[string]any, key string, value any) error {
	merged, err := merge(body[key], value, key)
	if err != nil {
		return err
	}
	body[key] = merged
	return nil
}

func merge(dst, src any, path string) (any, error) {
	dst, src = expand(dst), expand(src)
	if dst == nil {
		return src, nil
	}
	switch d := dst.(type) {
	case map[string]any:
		if s, ok := src.(map[string]any); ok {
			out := maps.Clone(d)
			for _, key := range slices.Sorted(maps.Keys(s)) {
				v, err := merge(d[key], s[key], path+"."+key)
				if err != nil {
					return nil, err
				}
				out[key] = v
			}
			return out, nil
		}
	case []any:
		if s, ok := src.([]any); ok {
			return append(slices.Clone(d), s...), nil
		}
	}
	return nil, fmt.Errorf("provider option %q conflicts with a generated request field", path)
}

// expand decodes one level of encoded JSON so an option can merge into it.
// Nested values stay encoded, keeping their bytes and key order.
func expand(v any) any {
	raw, ok := v.(json.RawMessage)
	if !ok {
		return v
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return v
	}
	switch raw[0] {
	case 'n':
		return nil
	case '{':
		var object map[string]json.RawMessage
		if json.Unmarshal(raw, &object) == nil {
			out := make(map[string]any, len(object))
			for key, value := range object {
				out[key] = value
			}
			return out
		}
	case '[':
		var array []json.RawMessage
		if json.Unmarshal(raw, &array) == nil {
			out := make([]any, len(array))
			for i, value := range array {
				out[i] = value
			}
			return out
		}
	}
	return v
}

// ApplyOptions puts options, in sorted key order, into body.
func ApplyOptions(body map[string]any, options map[string]any) error {
	for _, key := range slices.Sorted(maps.Keys(options)) {
		if err := PutOption(body, key, options[key]); err != nil {
			return err
		}
	}
	return nil
}

// MarshalBody encodes req, a typed request body, with options applied. Only
// the objects an option merges into are decoded; the rest of the body is
// copied as encoded.
func MarshalBody(req any, options map[string]any) ([]byte, error) {
	data, err := json.Marshal(req)
	if err != nil || len(options) == 0 {
		return data, err
	}
	body, ok := expand(json.RawMessage(data)).(map[string]any)
	if !ok {
		return nil, fmt.Errorf("request body is not a JSON object")
	}
	if err := ApplyOptions(body, options); err != nil {
		return nil, err
	}
	return json.Marshal(body)
}
