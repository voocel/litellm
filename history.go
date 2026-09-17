package litellm

import (
	"fmt"
	"regexp"
)

// ValidateHistory checks a complete, closed conversation: tool IDs are unique,
// every result references a pending call, and all calls are answered before the
// next non-tool message or the end of history. Client deliberately does not call
// it: partial histories and provider-managed state need not be self-contained.
// This check does not enforce any provider's ID spelling or message protocol.
func ValidateHistory(messages []Message) error {
	if err := validateMessages(messages); err != nil {
		return err
	}
	seen := make(map[string]bool)
	pending := make(map[string]bool)
	for i, msg := range messages {
		if msg.Role != RoleTool && len(pending) > 0 {
			return NewError(ErrorTypeValidation, fmt.Sprintf("messages[%d]: %s message follows unresolved tool use", i, msg.Role))
		}
		for _, block := range msg.Blocks {
			switch b := block.(type) {
			case ToolUseBlock:
				if seen[b.ID] {
					return NewError(ErrorTypeValidation, fmt.Sprintf("messages[%d]: duplicate tool use id %q", i, b.ID))
				}
				seen[b.ID] = true
				pending[b.ID] = true
			case ToolResultBlock:
				if !pending[b.ToolUseID] {
					return NewError(ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool result references unknown or already answered tool use %q", i, b.ToolUseID))
				}
				delete(pending, b.ToolUseID)
			}
		}
	}
	if len(pending) > 0 {
		return NewError(ErrorTypeValidation, "messages: unresolved tool use at end of history")
	}
	return nil
}

// MessageRepairPolicy selects explicit, potentially lossy history transformations.
// RepairMessages reports each change; Client never applies these policies.
type MessageRepairPolicy uint

const (
	RepairNone MessageRepairPolicy = 0

	RepairNormalizeToolUseIDs MessageRepairPolicy = 1 << iota
	RepairSynthesizeMissingToolUseIDs
	RepairInsertMissingToolResults

	RepairToolUseIDs = RepairNormalizeToolUseIDs | RepairSynthesizeMissingToolUseIDs
	RepairAll        = RepairToolUseIDs | RepairInsertMissingToolResults
)

var toolUseIDForbidden = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

const maxToolUseIDLen = 64

// RepairMessages returns an isolated copy of messages with the selected repairs.
// ID normalization targets the portable [a-zA-Z0-9_-], 64-byte form; it is not a
// restriction of the core model. Synthetic results describe interrupted execution
// and are always marked as errors. Call ValidateHistory afterwards to detect
// ambiguities, such as duplicate original IDs, that cannot be repaired safely.
func RepairMessages(messages []Message, policy MessageRepairPolicy) ([]Message, []Warning) {
	messages = cloneMessages(messages)
	if policy == RepairNone {
		return messages, nil
	}
	used := make(map[string]bool)
	for _, msg := range messages {
		for _, block := range msg.Blocks {
			switch b := block.(type) {
			case ToolUseBlock:
				used[b.ID] = true
			case ToolResultBlock:
				used[b.ToolUseID] = true
			}
		}
	}
	allocate := func(base string) string {
		candidate := base
		for n := 1; used[candidate]; n++ {
			suffix := fmt.Sprintf("_%d", n)
			prefix := base
			if len(prefix)+len(suffix) > maxToolUseIDLen {
				prefix = prefix[:maxToolUseIDLen-len(suffix)]
			}
			candidate = prefix + suffix
		}
		used[candidate] = true
		return candidate
	}
	idMap := make(map[string]string)
	open := make(map[string]ToolUseBlock)
	var openOrder []string
	out := make([]Message, 0, len(messages)+2)
	var warnings []Warning

	flushMissing := func() {
		if len(open) == 0 {
			return
		}
		if policy&RepairInsertMissingToolResults == 0 {
			return
		}
		for _, id := range openOrder {
			if _, ok := open[id]; !ok {
				continue
			}
			out = append(out, ToolResult(id,
				Text("Tool execution was interrupted; no result available."),
			))
			block := out[len(out)-1].Blocks[0].(ToolResultBlock)
			block.IsError = true
			out[len(out)-1].Blocks[0] = block
			warnings = append(warnings, Warning{
				Code:    "message.synthetic_tool_result_inserted",
				Message: fmt.Sprintf("assistant tool use %q had no matching tool result; inserted synthetic error result", id),
			})
			delete(open, id)
		}
		openOrder = openOrder[:0]
	}

	for _, msg := range messages {
		if msg.Role != RoleTool {
			flushMissing()
		}
		msg.Blocks = repairBlocks(msg.Role, msg.Blocks, policy, idMap, open, &openOrder, &warnings, allocate)
		out = append(out, msg)
	}
	flushMissing()
	return out, warnings
}

func repairBlocks(role Role, blocks []Block, policy MessageRepairPolicy, idMap map[string]string, open map[string]ToolUseBlock, openOrder *[]string, warnings *[]Warning, allocate func(string) string) []Block {
	out := make([]Block, len(blocks))
	for i, block := range blocks {
		switch b := block.(type) {
		case ToolUseBlock:
			if role == RoleAssistant {
				b = repairToolUseBlock(b, policy, idMap, warnings, allocate)
				if b.ID != "" {
					if policy&RepairInsertMissingToolResults != 0 {
						if _, exists := open[b.ID]; !exists {
							*openOrder = append(*openOrder, b.ID)
						}
					}
					open[b.ID] = b
				}
			}
			out[i] = b
		case ToolResultBlock:
			b = repairToolResultBlock(b, policy, idMap, warnings, allocate)
			if b.ToolUseID != "" {
				delete(open, b.ToolUseID)
			}
			out[i] = b
		default:
			out[i] = block
		}
	}
	return out
}

func repairToolUseBlock(block ToolUseBlock, policy MessageRepairPolicy, idMap map[string]string, warnings *[]Warning, allocate func(string) string) ToolUseBlock {
	original := block.ID
	if original == "" && policy&RepairSynthesizeMissingToolUseIDs != 0 {
		block.ID = allocate("call_repaired")
		*warnings = append(*warnings, Warning{
			Code:    "message.tool_use_id_synthesized",
			Message: fmt.Sprintf("assistant tool use %q was missing id; generated %q", block.Name, block.ID),
		})
		return block
	}
	if original == "" || policy&RepairNormalizeToolUseIDs == 0 {
		return block
	}
	normalized := NormalizeToolUseID(original)
	if normalized != original {
		if mapped, ok := idMap[original]; ok {
			normalized = mapped
		} else {
			normalized = allocate(normalized)
			idMap[original] = normalized
		}
		block.ID = normalized
		*warnings = append(*warnings, Warning{
			Code:    "message.tool_use_id_normalized",
			Message: fmt.Sprintf("assistant tool use id %q was normalized to %q", original, normalized),
		})
	}
	return block
}

func repairToolResultBlock(block ToolResultBlock, policy MessageRepairPolicy, idMap map[string]string, warnings *[]Warning, allocate func(string) string) ToolResultBlock {
	original := block.ToolUseID
	if mapped := idMap[original]; mapped != "" {
		block.ToolUseID = mapped
		*warnings = append(*warnings, Warning{
			Code:    "message.tool_use_id_normalized",
			Message: fmt.Sprintf("tool result id %q was normalized to %q", original, mapped),
		})
		return block
	}
	if original == "" || policy&RepairNormalizeToolUseIDs == 0 {
		return block
	}
	normalized := NormalizeToolUseID(original)
	if normalized != original {
		normalized = allocate(normalized)
		idMap[original] = normalized
		block.ToolUseID = normalized
		*warnings = append(*warnings, Warning{
			Code:    "message.tool_use_id_normalized",
			Message: fmt.Sprintf("tool result id %q was normalized to %q", original, normalized),
		})
	}
	return block
}

// NormalizeToolUseID returns the portable spelling of an ID. Use RepairMessages
// to normalize a whole history without introducing collisions between IDs.
func NormalizeToolUseID(id string) string {
	out := toolUseIDForbidden.ReplaceAllString(id, "_")
	if len(out) > maxToolUseIDLen {
		out = out[:maxToolUseIDLen]
	}
	return out
}
