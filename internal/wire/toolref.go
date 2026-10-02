package wire

import "github.com/voocel/litellm"

// ToolReferenceText is a tool reference as text, for wire formats without
// one: the referenced tool is among the request's tools.
func ToolReferenceText(ref litellm.ToolReferenceBlock) string {
	return "Tool " + ref.ToolName + " is now available."
}
