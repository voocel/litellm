package openaicompat

import (
	"slices"

	"github.com/voocel/litellm"
)

func (s Spec) usesSchemaPrompt(format *litellm.ResponseFormat) bool {
	return s.SchemaFallback != "" && format != nil && format.Type == litellm.ResponseFormatJSONSchema
}

func (s Spec) schemaWarning(provider string) litellm.Warning {
	mode := "prompt only"
	if s.SchemaFallback == litellm.ResponseFormatJSONObject {
		mode = "prompt with JSON mode"
	}
	return litellm.Warning{
		Code:     "litellm.schema_prompt_fallback",
		Provider: provider,
		Message:  "JSON Schema uses " + mode + "; schema adherence is not enforced, even with Strict",
	}
}

func withSchemaPrompt(messages []litellm.Message, schema *litellm.JSONSchema) []litellm.Message {
	prompt := "\n\nFor your final answer, return only JSON matching the following schema, without Markdown fences or extra text.\nSchema: " + schema.Name
	if schema.Description != "" {
		prompt += "\nDescription: " + schema.Description
	}
	prompt += "\n" + string(schema.Schema)

	out := slices.Clone(messages)
	// Attach to the last user turn to preserve tool-call/result ordering and
	// reasoning replay. Copy its blocks as well so repeated calls do not append
	// instructions to the caller's history or overwrite spare slice capacity.
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == litellm.RoleUser {
			out[i].Blocks = append(slices.Clone(out[i].Blocks), litellm.Text(prompt))
			return out
		}
	}
	return append(out, litellm.UserText(prompt))
}
