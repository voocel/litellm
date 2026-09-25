package litellm

import (
	"encoding/json"
	"fmt"
	"math"
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
				if !utf8.ValidString(b.Signature) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: text block signature must be valid UTF-8", i), nil)
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
				if !utf8.ValidString(b.Signature) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: reasoning block signature must be valid UTF-8", i), nil)
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
				if !utf8.ValidString(b.Signature) {
					return NewError("", ErrorTypeValidation, fmt.Sprintf("messages[%d]: tool use %q signature must be valid UTF-8", i, b.ID), nil)
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
				return NewError(resolvedProvider, ErrorTypeValidation, "tool use missing id", nil)
			}
			if tool.Name == "" {
				return NewError(resolvedProvider, ErrorTypeValidation, fmt.Sprintf("tool use %q missing name", tool.ID), nil)
			}
		}
	}
	return nil
}

func finalizeResponse(resp *Response, provider, model string) {
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
// JSON. Raw arguments stay out of warnings so observers never receive large or
// sensitive payloads.
func malformedToolArgumentWarnings(blocks []Block, provider string) []Warning {
	var warnings []Warning
	for _, block := range blocks {
		tool, ok := block.(ToolUseBlock)
		if !ok || len(tool.Arguments) == 0 || json.Valid(tool.Arguments) {
			continue
		}
		var probe any
		parseErr := json.Unmarshal(tool.Arguments, &probe)
		warnings = append(warnings, Warning{
			Code:     "litellm.tool_arguments_invalid",
			Provider: provider,
			Message:  fmt.Sprintf("tool use %q returned malformed JSON arguments: %v", tool.ID, parseErr),
		})
	}
	return warnings
}

func cloneRequest(req Request) *Request {
	out := req
	out.MaxTokens = clonePtr(req.MaxTokens)
	out.Temperature = clonePtr(req.Temperature)
	out.TopP = clonePtr(req.TopP)
	out.Messages = cloneMessages(req.Messages)
	out.Stop = append([]string(nil), req.Stop...)
	out.Tools = cloneTools(req.Tools)
	if req.ToolChoice != nil {
		choice := *req.ToolChoice
		out.ToolChoice = &choice
	}
	out.ResponseFormat = cloneResponseFormat(req.ResponseFormat)
	out.Thinking = cloneThinking(req.Thinking)
	if req.ProviderOptions != nil {
		out.ProviderOptions = make(ProviderOptions, len(req.ProviderOptions))
		for k, v := range req.ProviderOptions {
			out.ProviderOptions[k] = append(json.RawMessage(nil), v...)
		}
	}
	return &out
}

func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]Message, len(messages))
	for i, msg := range messages {
		out[i] = Message{Role: msg.Role, Blocks: cloneBlocks(msg.Blocks)}
	}
	return out
}

func cloneBlocks(blocks []Block) []Block {
	if len(blocks) == 0 {
		return nil
	}
	out := make([]Block, len(blocks))
	for i, block := range blocks {
		out[i] = cloneBlock(block)
	}
	return out
}

func cloneBlock(block Block) Block {
	switch b := block.(type) {
	case TextBlock:
		b.Logprobs = cloneBytes(b.Logprobs)
		b.Annotations = append([]Annotation(nil), b.Annotations...)
		for i := range b.Annotations {
			b.Annotations[i].Extra = cloneBytes(b.Annotations[i].Extra)
		}
		b.Cache = cloneCacheControl(b.Cache)
		return b
	case ImageBlock:
		b.Data = cloneBytes(b.Data)
		b.Cache = cloneCacheControl(b.Cache)
		return b
	case ReasoningBlock:
		b.Redacted = cloneBytes(b.Redacted)
		b.Extra = cloneBytes(b.Extra)
		b.Cache = cloneCacheControl(b.Cache)
		return b
	case ToolUseBlock:
		b.Arguments = cloneBytes(b.Arguments)
		b.Cache = cloneCacheControl(b.Cache)
		return b
	case ToolResultBlock:
		b.Content = cloneBlocks(b.Content)
		b.Cache = cloneCacheControl(b.Cache)
		return b
	case ToolReferenceBlock:
		b.Cache = cloneCacheControl(b.Cache)
		return b
	default:
		return block
	}
}

func cloneResponse(resp *Response) *Response {
	if resp == nil {
		return nil
	}
	out := *resp
	out.Blocks = cloneBlocks(resp.Blocks)
	out.Usage = resp.Usage.Clone()
	out.Warnings = append([]Warning(nil), resp.Warnings...)
	out.Raw = cloneBytes(resp.Raw)
	return &out
}

func cloneTools(tools []Tool) []Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]Tool, len(tools))
	for i, tool := range tools {
		out[i] = tool
		out[i].Parameters = Schema(cloneBytes(tool.Parameters))
	}
	return out
}

func clonePtr[T any](v *T) *T {
	if v == nil {
		return nil
	}
	return new(*v)
}

func cloneResponseFormat(format *ResponseFormat) *ResponseFormat {
	if format == nil {
		return nil
	}
	out := *format
	if format.JSONSchema != nil {
		schema := *format.JSONSchema
		schema.Schema = Schema(cloneBytes(format.JSONSchema.Schema))
		out.JSONSchema = &schema
	}
	return &out
}

func cloneThinking(thinking *Thinking) *Thinking {
	if thinking == nil {
		return nil
	}
	out := *thinking
	out.BudgetTokens = clonePtr(thinking.BudgetTokens)
	return &out
}

func cloneCacheControl(cache *CacheControl) *CacheControl {
	if cache == nil {
		return nil
	}
	out := *cache
	return &out
}
