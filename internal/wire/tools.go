package wire

import (
	"encoding/json"
	"fmt"

	"github.com/voocel/litellm"
)

// ToolParameters returns the parameters schema of t, one taking no
// arguments when it has none, as wire formats require a schema.
func ToolParameters(t litellm.Tool) json.RawMessage {
	if len(t.Parameters) == 0 {
		return json.RawMessage(`{"type":"object"}`)
	}
	return json.RawMessage(t.Parameters)
}

// ToolInput returns the arguments of call as the JSON object wire formats
// such as Anthropic's, Bedrock's and Gemini's need, the empty object when it
// has none. Arguments that are not an object fail, naming the call.
func ToolInput(call litellm.ToolUseBlock) (json.RawMessage, error) {
	if call.Arguments == "" {
		return json.RawMessage("{}"), nil
	}
	if !IsObject(call.Arguments) {
		return nil, fmt.Errorf("tool use %q (%s) arguments are not a JSON object", call.ID, call.Name)
	}
	return json.RawMessage(call.Arguments), nil
}

// IsObject reports whether text is a JSON object.
func IsObject(text string) bool {
	var object map[string]json.RawMessage
	return json.Unmarshal([]byte(text), &object) == nil && object != nil
}

// ToolReferenceText is a tool reference as text, for wire formats without
// one: the referenced tool is among the request's tools.
func ToolReferenceText(ref litellm.ToolReferenceBlock) string {
	return "Tool " + ref.ToolName + " is now available."
}
