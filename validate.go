package litellm

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"unicode/utf8"
)

func validateRequest(req *Request) error {
	if req == nil {
		return NewError("", ErrorTypeValidation, "request cannot be nil", nil)
	}
	if req.Model == "" {
		return NewError("", ErrorTypeValidation, "model cannot be empty", nil)
	}
	if !utf8.ValidString(req.Model) {
		return NewError("", ErrorTypeValidation, "model must be valid UTF-8", nil)
	}
	if len(req.Messages) == 0 {
		return NewError("", ErrorTypeValidation, "messages cannot be empty", nil)
	}
	for i, stop := range req.Stop {
		if !utf8.ValidString(stop) {
			return NewError("", ErrorTypeValidation, fmt.Sprintf("stop[%d] must be valid UTF-8", i), nil)
		}
	}
	if req.MaxTokens != nil && *req.MaxTokens <= 0 {
		return NewError("", ErrorTypeValidation, "max_tokens must be positive", nil)
	}
	if req.Temperature != nil && (math.IsNaN(*req.Temperature) || math.IsInf(*req.Temperature, 0)) {
		return NewError("", ErrorTypeValidation, "temperature must be finite", nil)
	}
	if req.TopP != nil && (math.IsNaN(*req.TopP) || math.IsInf(*req.TopP, 0)) {
		return NewError("", ErrorTypeValidation, "top_p must be finite", nil)
	}
	if err := validateMessages(req.Messages); err != nil {
		return err
	}
	for _, tool := range req.Tools {
		if tool.Name == "" {
			return NewError("", ErrorTypeValidation, "tool name cannot be empty", nil)
		}
		if !utf8.ValidString(tool.Name) {
			return NewError("", ErrorTypeValidation, "tool name must be valid UTF-8", nil)
		}
		if !utf8.ValidString(tool.Description) {
			return NewError("", ErrorTypeValidation, fmt.Sprintf("tool %q description must be valid UTF-8", tool.Name), nil)
		}
		if len(tool.Parameters) > 0 && !json.Valid(tool.Parameters) {
			return NewError("", ErrorTypeValidation, fmt.Sprintf("tool %q parameters must be valid JSON", tool.Name), nil)
		}
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type == ResponseFormatJSONSchema {
		if req.ResponseFormat.JSONSchema == nil {
			return NewError("", ErrorTypeValidation, "json schema response format requires schema", nil)
		}
		if req.ResponseFormat.JSONSchema.Name == "" {
			return NewError("", ErrorTypeValidation, "json schema response format requires name", nil)
		}
		if !utf8.ValidString(req.ResponseFormat.JSONSchema.Name) {
			return NewError("", ErrorTypeValidation, "json schema response format name must be valid UTF-8", nil)
		}
		if !utf8.ValidString(req.ResponseFormat.JSONSchema.Description) {
			return NewError("", ErrorTypeValidation, "json schema response format description must be valid UTF-8", nil)
		}
		if len(req.ResponseFormat.JSONSchema.Schema) > 0 && !json.Valid(req.ResponseFormat.JSONSchema.Schema) {
			return NewError("", ErrorTypeValidation, "json schema response format schema must be valid JSON", nil)
		}
	}
	if err := req.Thinking.validate(); err != nil {
		return err
	}
	if err := req.ToolChoice.validate(); err != nil {
		return err
	}
	if err := req.ProviderOptions.validate(); err != nil {
		return err
	}
	return nil
}

func validateMessages(messages []Message) error {
	for i, msg := range messages {
		switch msg.Role {
		case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		default:
			return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: invalid role %q", i, msg.Role), nil)
		}
		for _, block := range msg.Blocks {
			switch b := block.(type) {
			case TextBlock:
				if !utf8.ValidString(b.Text) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: text block must be valid UTF-8", i), nil)
				}
				if err := validateState(i, b.State); err != nil {
					return err
				}
			case ImageBlock:
				if err := validateImageUTF8(i, b); err != nil {
					return err
				}
			case ReasoningBlock:
				if msg.Role != RoleAssistant {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: reasoning block requires assistant role", i), nil)
				}
				if !utf8.ValidString(b.Text) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: reasoning block text must be valid UTF-8", i), nil)
				}
				if err := validateState(i, b.State); err != nil {
					return err
				}
			case ToolReferenceBlock:
				return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool reference block is only valid inside tool result content", i), nil)
			case ToolUseBlock:
				if msg.Role != RoleAssistant {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use block requires assistant role", i), nil)
				}
				if b.ID == "" {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use missing id", i), nil)
				}
				if b.Name == "" {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use %q missing name", i, b.ID), nil)
				}
				if !utf8.ValidString(b.ID) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use id must be valid UTF-8", i), nil)
				}
				if !utf8.ValidString(b.Name) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use %q name must be valid UTF-8", i, b.ID), nil)
				}
				if err := validateState(i, b.State); err != nil {
					return err
				}
			case ToolResultBlock:
				if msg.Role != RoleTool {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool result block requires tool role", i), nil)
				}
				if b.ToolUseID == "" {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool result missing tool use id", i), nil)
				}
				if !utf8.ValidString(b.ToolUseID) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool result id must be valid UTF-8", i), nil)
				}
				if err := validateToolResultContent(i, b.Content); err != nil {
					return err
				}
			default:
				return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: unsupported block %T", i, block), nil)
			}
		}
	}
	return nil
}

func validateState(messageIndex int, state *ProviderState) error {
	switch {
	case state == nil:
		return nil
	case state.Provider == "":
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: provider state missing provider", messageIndex), nil)
	case !utf8.ValidString(state.Provider) || !utf8.ValidString(state.Model):
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: provider state must be valid UTF-8", messageIndex), nil)
	case !json.Valid(state.Data):
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: provider state data must be valid JSON", messageIndex), nil)
	}
	return nil
}

func validateToolResultContent(messageIndex int, blocks []Block) error {
	for _, block := range blocks {
		switch b := block.(type) {
		case TextBlock:
			if !utf8.ValidString(b.Text) {
				return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool result text block must be valid UTF-8", messageIndex), nil)
			}
		case ImageBlock:
			if err := validateImageUTF8(messageIndex, b); err != nil {
				return err
			}
		case ToolReferenceBlock:
			if b.ToolName == "" {
				return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool reference missing tool name", messageIndex), nil)
			}
			if !utf8.ValidString(b.ToolName) {
				return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool reference name must be valid UTF-8", messageIndex), nil)
			}
		default:
			return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: unsupported tool result content block %T", messageIndex, block), nil)
		}
	}
	return nil
}

func validateImageUTF8(messageIndex int, block ImageBlock) error {
	if !utf8.ValidString(block.URL) {
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: image URL must be valid UTF-8", messageIndex), nil)
	}
	if !utf8.ValidString(block.MIME) {
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: image MIME must be valid UTF-8", messageIndex), nil)
	}
	if !utf8.ValidString(block.FileURI) {
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: image file URI must be valid UTF-8", messageIndex), nil)
	}
	if !utf8.ValidString(block.Detail) {
		return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: image detail must be valid UTF-8", messageIndex), nil)
	}
	return nil
}

// validateResponse checks a reply: a tool call the vendor left without an id
// or a name is a provider error.
func validateResponse(resp *Response, provider, model string) error {
	if resp == nil {
		return NewError(provider, ErrorTypeInternal, "provider returned nil response without error", nil)
	}
	resolvedProvider := resp.Provider
	if resolvedProvider == "" {
		resolvedProvider = provider
	}
	resolvedModel := resp.Model
	if resolvedModel == "" {
		resolvedModel = model
	}
	if resolvedProvider == "" {
		return NewError(provider, ErrorTypeInternal, "response missing provider", nil)
	}
	if resolvedModel == "" {
		return NewError(resolvedProvider, ErrorTypeInternal, "response missing model", nil)
	}
	for _, block := range resp.Blocks {
		if tool, ok := block.(ToolUseBlock); ok {
			if tool.ID == "" {
				return NewError(resolvedProvider, ErrorTypeProvider, "tool use missing id", nil)
			}
			if tool.Name == "" {
				return NewError(resolvedProvider, ErrorTypeProvider, fmt.Sprintf("tool use %q missing name", tool.ID), nil)
			}
		}
	}
	return nil
}

func finalizeResponse(resp *Response, provider, model string) {
	resp.FinishReason = finishReason(resp.FinishReason, resp.Blocks)
	if resp.Provider == "" {
		resp.Provider = provider
	}
	if resp.Model == "" {
		resp.Model = model
	}
	for i := range resp.Warnings {
		if resp.Warnings[i].Provider == "" {
			resp.Warnings[i].Provider = resp.Provider
		}
	}
	resp.Warnings = append(resp.Warnings, malformedToolArgumentWarnings(resp.Blocks, resp.Provider)...)
}

// malformedToolArgumentWarnings reports tool calls whose arguments are not valid
// JSON; the Client adds them to a response once. Raw arguments stay out of
// warnings so observers never receive large or sensitive payloads.
func malformedToolArgumentWarnings(blocks []Block, provider string) []Warning {
	var warnings []Warning
	for _, block := range blocks {
		tool, ok := block.(ToolUseBlock)
		if !ok || tool.Arguments == "" || json.Valid([]byte(tool.Arguments)) {
			continue
		}
		var probe any
		parseErr := json.Unmarshal([]byte(tool.Arguments), &probe)
		warnings = append(warnings, Warning{
			Code:     "litellm.tool_arguments_invalid",
			Provider: provider,
			Message:  fmt.Sprintf("tool use %q returned malformed JSON arguments: %v", tool.ID, parseErr),
		})
	}
	return warnings
}

// finishReason is why a reply ended: one that stopped with tool calls, as
// Gemini, OpenAI Responses and some Chat Completions vendors end a turn of
// calls, ended for them.
func finishReason(reason FinishReason, blocks []Block) FinishReason {
	isCall := func(b Block) bool { _, ok := b.(ToolUseBlock); return ok }
	if reason == FinishReasonStop && slices.ContainsFunc(blocks, isCall) {
		return FinishReasonToolCall
	}
	return reason
}
