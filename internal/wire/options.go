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

// ApplyOptions puts options, in sorted key order, into body.
func ApplyOptions(body map[string]any, options map[string]any) error {
	for _, key := range slices.Sorted(maps.Keys(options)) {
		if err := PutOption(body, key, options[key]); err != nil {
			return err
		}
	}
	return nil
}

// MarshalBody encodes req, a typed request body, with options applied.
func MarshalBody(req any, options map[string]any) ([]byte, error) {
	data, err := json.Marshal(req)
	if err != nil || len(options) == 0 {
		return data, err
	}
	body := make(map[string]any)
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&body); err != nil {
		return nil, err
	}
	if err := ApplyOptions(body, options); err != nil {
		return nil, err
	}
	return json.Marshal(body)
}
