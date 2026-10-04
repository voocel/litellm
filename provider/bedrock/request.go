package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/claude"
	"github.com/voocel/litellm/internal/wire"
)

// ProviderOptions are native Converse fields copied into the body. An option
// naming a generated object is merged into it, so model-specific fields go
// under "additionalModelRequestFields".
const (
	ProviderOptionAdditionalModelRequestFields      = "additionalModelRequestFields"
	ProviderOptionAdditionalModelResponseFieldPaths = "additionalModelResponseFieldPaths"
	ProviderOptionGuardrailConfig                   = "guardrailConfig"
	ProviderOptionPerformanceConfig                 = "performanceConfig"
	ProviderOptionPromptVariables                   = "promptVariables"
	ProviderOptionRequestMetadata                   = "requestMetadata"
)

var providerOptions = []string{
	ProviderOptionAdditionalModelRequestFields, ProviderOptionAdditionalModelResponseFieldPaths,
	ProviderOptionGuardrailConfig, ProviderOptionPerformanceConfig, ProviderOptionPromptVariables,
	ProviderOptionRequestMetadata,
}

func sortedOptions() []string {
	out := slices.Clone(providerOptions)
	slices.Sort(out)
	return out
}

// buildRequest maps Thinking in Anthropic's format through
// additionalModelRequestFields; other model families set their own fields
// with that option.
func buildRequest(req *litellm.Request, provider string) ([]byte, error) {
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if err := wire.CheckOptions(opts, providerOptions); err != nil {
		return nil, err
	}
	out := &request{InferenceConfig: convertInference(req)}
	if err := convertMessages(out, req.Messages, provider, req.Model); err != nil {
		return nil, err
	}
	if thinking := claude.Thinking(req.Thinking); thinking != nil {
		out.AdditionalModelRequestFields = map[string]any{"thinking": thinking}
		if thinking.Effort != "" {
			out.AdditionalModelRequestFields["output_config"] = map[string]any{"effort": thinking.Effort}
		}
	}
	if out.OutputConfig, err = convertOutputConfig(req.ResponseFormat); err != nil {
		return nil, err
	}
	if len(req.Tools) > 0 {
		if out.ToolConfig, err = convertToolConfig(req); err != nil {
			return nil, err
		}
	}
	return wire.MarshalBody(out, opts)
}

// convertToolConfig expresses ToolChoiceNone by omitting the tools. Converse
// has no "none" choice and requires tools whenever history holds tool blocks.
// It cannot defer a tool either, and a tool list that changes invalidates
// Claude's thinking, so deferred tools are offered from the start.
func convertToolConfig(req *litellm.Request) (*toolConfig, error) {
	choice := req.ToolChoice
	if choice != nil && choice.Mode == litellm.ToolChoiceNone {
		if hasToolBlocks(req.Messages) {
			return nil, errors.New("tool_choice none cannot be expressed when history contains tool calls")
		}
		return nil, nil
	}
	out := &toolConfig{}
	for _, t := range req.Tools {
		spec := &toolSpec{Name: t.Name, Description: t.Description, InputSchema: inputSchema{JSON: json.RawMessage(`{"type":"object"}`)}}
		if len(t.Parameters) > 0 {
			spec.InputSchema.JSON = json.RawMessage(t.Parameters)
		}
		spec.Strict = t.Strict
		out.Tools = append(out.Tools, tool{ToolSpec: spec})
	}
	switch {
	case choice == nil:
	case choice.Name != "":
		out.ToolChoice = map[string]any{"tool": map[string]any{"name": choice.Name}}
	case choice.Mode == litellm.ToolChoiceRequired:
		out.ToolChoice = map[string]any{"any": map[string]any{}}
	default:
		out.ToolChoice = map[string]any{"auto": map[string]any{}}
	}
	return out, nil
}

func hasToolBlocks(messages []litellm.Message) bool {
	for _, msg := range messages {
		for _, block := range msg.Blocks {
			switch block.(type) {
			case litellm.ToolUseBlock, litellm.ToolResultBlock:
				return true
			}
		}
	}
	return false
}

func convertMessages(out *request, messages []litellm.Message, provider, model string) error {
	for i, msg := range messages {
		if msg.Role == litellm.RoleSystem {
			for _, block := range msg.Blocks {
				text, ok := block.(litellm.TextBlock)
				if !ok {
					return fmt.Errorf("messages[%d]: system only supports text blocks, got %T", i, block)
				}
				if text.Text == "" {
					continue // an empty text block has no content member
				}
				out.System = append(out.System, content{Text: text.Text})
				if text.Cache != nil {
					out.System = append(out.System, content{CachePoint: convertCache(text.Cache)})
				}
			}
			continue
		}
		blocks, err := convertBlocks(msg.Blocks, provider, model)
		if err != nil {
			return fmt.Errorf("messages[%d]: %w", i, err)
		}
		if len(blocks) == 0 {
			continue // left empty, such as by dropped foreign reasoning
		}
		role := "user"
		if msg.Role == litellm.RoleAssistant {
			role = "assistant"
		}
		// Roles must alternate, and parallel tool results share one user turn.
		if n := len(out.Messages); n > 0 && out.Messages[n-1].Role == role {
			out.Messages[n-1].Content = append(out.Messages[n-1].Content, blocks...)
			continue
		}
		out.Messages = append(out.Messages, message{Role: role, Content: blocks})
	}
	return nil
}

// convertBlocks maps blocks; a cache breakpoint becomes a cachePoint after
// its block.
func convertBlocks(blocks []litellm.Block, provider, model string) ([]content, error) {
	out := make([]content, 0, len(blocks))
	for _, block := range blocks {
		var c content
		var cache *litellm.CacheControl
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue // an empty text block has no content member
			}
			c, cache = content{Text: b.Text}, b.Cache
		case litellm.ImageBlock:
			img, err := convertImage(b)
			if err != nil {
				return nil, err
			}
			c, cache = content{Image: img}, b.Cache
		case litellm.ReasoningBlock:
			// Reasoning from elsewhere lacks the signature models require,
			// including that of another model: one provider serves models of
			// every family, which reject each other's reasoning.
			state, ok := wire.ReadState[reasoningState](b.State, provider)
			if !ok || b.State.Model != model {
				continue
			}
			c = content{ReasoningContent: &reasoningContent{ReasoningText: &reasoningText{Text: b.Text, Signature: state.Signature}}}
			if len(state.RedactedContent) > 0 {
				c.ReasoningContent = &reasoningContent{RedactedContent: state.RedactedContent}
			}
		case litellm.ToolUseBlock:
			input := json.RawMessage("{}")
			if b.Arguments != "" {
				var object map[string]json.RawMessage
				if json.Unmarshal([]byte(b.Arguments), &object) != nil || object == nil {
					return nil, fmt.Errorf("tool use %q (%s) arguments are not a JSON object", b.ID, b.Name)
				}
				input = json.RawMessage(b.Arguments)
			}
			c, cache = content{ToolUse: &toolUse{ToolUseID: claude.ToolUseID(b.ID), Name: b.Name, Input: input}}, b.Cache
		case litellm.ToolResultBlock:
			result := &toolResult{ToolUseID: claude.ToolUseID(b.ToolUseID), Content: make([]content, 0, len(b.Content))} // content is required, even empty
			if b.IsError {
				result.Status = "error"
			}
			for _, child := range b.Content {
				switch child := child.(type) {
				case litellm.TextBlock:
					if child.Text == "" {
						continue // as for message text
					}
					result.Content = append(result.Content, content{Text: child.Text})
				case litellm.ImageBlock:
					img, err := convertImage(child)
					if err != nil {
						return nil, err
					}
					result.Content = append(result.Content, content{Image: img})
				case litellm.ToolReferenceBlock:
					result.Content = append(result.Content, content{Text: wire.ToolReferenceText(child)})
				default:
					return nil, fmt.Errorf("unsupported tool result content %T", child)
				}
			}
			c, cache = content{ToolResult: result}, b.Cache
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
		out = append(out, c)
		if cache != nil {
			out = append(out, content{CachePoint: convertCache(cache)})
		}
	}
	return out, nil
}

func convertCache(cache *litellm.CacheControl) *cachePoint {
	return &cachePoint{Type: "default", TTL: cache.TTL}
}

// convertImage sends bytes; Converse takes a data URL's payload but not
// remote URLs.
func convertImage(block litellm.ImageBlock) (*image, error) {
	mime, data := block.MIME, block.Data
	if len(data) == 0 {
		dataMIME, encoded, ok := wire.ParseDataURL(block.URL)
		if !ok {
			return nil, errors.New("image requires inline data or a data URL")
		}
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("image data URL must be base64: %w", err)
		}
		mime, data = dataMIME, decoded
	}
	format, ok := strings.CutPrefix(mime, "image/")
	if !ok || format == "" {
		return nil, fmt.Errorf("image MIME %q must be image/<format>", mime)
	}
	return &image{Format: format, Source: imageSource{Bytes: data}}, nil
}

func convertInference(req *litellm.Request) *inferenceConfig {
	if req.MaxTokens == nil && req.Temperature == nil && req.TopP == nil && len(req.Stop) == 0 {
		return nil
	}
	return &inferenceConfig{MaxTokens: req.MaxTokens, Temperature: req.Temperature, TopP: req.TopP, StopSequences: req.Stop}
}

func convertOutputConfig(format *litellm.ResponseFormat) (*outputConfig, error) {
	if format == nil {
		return nil, nil
	}
	switch format.Type {
	case "", litellm.ResponseFormatText:
		return nil, nil
	case litellm.ResponseFormatJSONSchema:
		schema := jsonSchema{Name: format.JSONSchema.Name, Description: format.JSONSchema.Description, Schema: string(format.JSONSchema.Schema)}
		return &outputConfig{TextFormat: &textFormat{Type: "json_schema", Structure: textFormatStructure{JSONSchema: schema}}}, nil
	case litellm.ResponseFormatJSONObject:
		return nil, errors.New("response_format json_object has no Converse equivalent; use json_schema")
	default:
		return nil, fmt.Errorf("unsupported response format %q", format.Type)
	}
}
