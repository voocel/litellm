package anthropic

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/voocel/litellm"
)

type anthropicRequest struct {
	Model         string                 `json:"model"`
	System        any                    `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Messages      []anthropicMessage     `json:"messages"`
	Stream        bool                   `json:"stream,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	Tools         []anthropicTool        `json:"tools,omitempty"`
	ToolChoice    any                    `json:"tool_choice,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Thinking      *anthropicThinking     `json:"thinking,omitempty"`
	OutputConfig  *anthropicOutputConfig `json:"output_config,omitempty"`
	Metadata      map[string]any         `json:"metadata,omitempty"`
}

type anthropicMessage struct {
	Role    string             `json:"role"`
	Content []anthropicContent `json:"content"`
}

type anthropicContent struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	Source       *anthropicImageSource  `json:"source,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`
	Signature    string                 `json:"signature,omitempty"`
	Data         string                 `json:"data,omitempty"`
	ID           string                 `json:"id,omitempty"`
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Input        *map[string]any        `json:"input,omitempty"`
	Content      any                    `json:"content,omitempty"`
	ToolName     string                 `json:"tool_name,omitempty"`
	IsError      bool                   `json:"is_error,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
	Strict      *bool          `json:"strict,omitempty"`
}

type anthropicThinking struct {
	Type    string `json:"type"`
	Display string `json:"display,omitempty"`
}

type anthropicOutputConfig struct {
	Effort string                 `json:"effort,omitempty"`
	Format *anthropicOutputFormat `json:"format,omitempty"`
}

type anthropicOutputFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

func warning(code, message string) litellm.Warning {
	return litellm.Warning{Code: code, Provider: "anthropic", Message: message}
}

const (
	ProviderOptionMetadata       = "metadata"
	ProviderOptionMetadataUserID = "metadata_user_id"
)

func (p *Provider) buildRequest(req *litellm.Request, stream bool) (*anthropicRequest, []litellm.Warning, error) {
	if req.MaxTokens == nil {
		return nil, nil, fmt.Errorf("anthropic: max_tokens is required")
	}
	if req.Temperature != nil && req.TopP != nil {
		return nil, nil, fmt.Errorf("anthropic: temperature and top_p cannot both be set")
	}
	metadata, err := anthropicMetadata(req.ProviderOptions)
	if err != nil {
		return nil, nil, err
	}
	temperature, topP := req.Temperature, req.TopP
	if err := validateSampling(temperature, topP); err != nil {
		return nil, nil, err
	}
	toolChoice, err := convertToolChoice(req.ToolChoice)
	if err != nil {
		return nil, nil, err
	}
	out := &anthropicRequest{
		Model:         req.Model,
		MaxTokens:     *req.MaxTokens,
		Stream:        stream,
		Temperature:   temperature,
		TopP:          topP,
		StopSequences: append([]string(nil), req.Stop...),
		ToolChoice:    toolChoice,
		Metadata:      metadata,
	}
	thinking, effort, err := convertThinking(req.Thinking)
	if err != nil {
		return nil, nil, err
	}
	out.Thinking = thinking
	format, err := convertResponseFormat(req.ResponseFormat)
	if err != nil {
		return nil, nil, err
	}
	if effort != "" || format != nil {
		out.OutputConfig = &anthropicOutputConfig{Effort: effort, Format: format}
	}
	if len(req.Tools) > 0 {
		out.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			converted, err := convertTool(tool)
			if err != nil {
				return nil, nil, err
			}
			out.Tools = append(out.Tools, converted)
		}
	}
	system, messages, err := convertMessages(req.Messages)
	if err != nil {
		return nil, nil, err
	}
	out.System = system
	out.Messages = messages
	return out, nil, nil
}

func convertToolChoice(choice *litellm.ToolChoice) (any, error) {
	if err := choice.Validate(); err != nil {
		return nil, err
	}
	if choice == nil {
		return nil, nil
	}
	if choice.Name != "" {
		return map[string]any{"type": "tool", "name": choice.Name}, nil
	}
	mode := string(choice.Mode)
	if choice.Mode == litellm.ToolChoiceRequired {
		mode = "any"
	}
	return map[string]any{"type": mode}, nil
}

func convertResponseFormat(format *litellm.ResponseFormat) (*anthropicOutputFormat, error) {
	if format == nil {
		return nil, nil
	}
	switch format.Type {
	case "", litellm.ResponseFormatText:
		return nil, nil
	case litellm.ResponseFormatJSONSchema:
		if format.JSONSchema == nil || len(format.JSONSchema.Schema) == 0 {
			return nil, fmt.Errorf("anthropic: response_format json_schema requires a schema")
		}
		return &anthropicOutputFormat{Type: "json_schema", Schema: json.RawMessage(format.JSONSchema.Schema)}, nil
	case litellm.ResponseFormatJSONObject:
		return nil, fmt.Errorf("anthropic: response_format json_object is not supported; use json_schema")
	default:
		return nil, fmt.Errorf("anthropic: unsupported response format %q", format.Type)
	}
}

func anthropicMetadata(rawOptions litellm.ProviderOptions) (map[string]any, error) {
	options, err := rawOptions.Decode()
	if err != nil {
		return nil, err
	}

	if len(options) == 0 {
		return nil, nil
	}
	for key := range options {
		switch key {
		case ProviderOptionMetadata, ProviderOptionMetadataUserID:
		default:
			return nil, fmt.Errorf("anthropic: unsupported provider option %q", key)
		}
	}
	var metadata map[string]any
	if raw, ok := options[ProviderOptionMetadata]; ok && raw != nil {
		switch value := raw.(type) {
		case map[string]any:
			metadata = make(map[string]any, len(value))
			for k, v := range value {
				if k == "" {
					return nil, fmt.Errorf("anthropic: metadata key cannot be empty")
				}
				metadata[k] = v
			}
		case map[string]string:
			metadata = make(map[string]any, len(value))
			for k, v := range value {
				if k == "" {
					return nil, fmt.Errorf("anthropic: metadata key cannot be empty")
				}
				metadata[k] = v
			}
		default:
			return nil, fmt.Errorf("anthropic: provider option %q must be object", "metadata")
		}
	}
	if raw, ok := options[ProviderOptionMetadataUserID]; ok && raw != nil {
		userID, ok := raw.(string)
		if !ok {
			return nil, fmt.Errorf("anthropic: provider option %q must be string", "metadata_user_id")
		}
		if userID != "" {
			if metadata == nil {
				metadata = map[string]any{}
			}
			metadata["user_id"] = userID
		}
	}
	if len(metadata) == 0 {
		return nil, nil
	}
	return metadata, nil
}

func validateSampling(temperature, topP *float64) error {
	if temperature != nil && *temperature != 1 {
		return fmt.Errorf("anthropic: temperature must be 1 on current Claude models, got %g", *temperature)
	}
	if topP != nil && (math.IsNaN(*topP) || *topP < 0.99 || *topP > 1) {
		return fmt.Errorf("anthropic: top_p must be between 0.99 and 1 on current Claude models, got %g", *topP)
	}
	return nil
}

func convertThinking(thinking *litellm.Thinking) (*anthropicThinking, string, error) {
	if err := thinking.Validate(); err != nil {
		return nil, "", fmt.Errorf("anthropic: %w", err)
	}
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, "", nil
	}
	if thinking.Mode == litellm.ThinkingDisabled {
		return &anthropicThinking{Type: "disabled"}, "", nil
	}
	if thinking.BudgetTokens != nil {
		return nil, "", fmt.Errorf("anthropic: budget_tokens is not supported; use effort with adaptive thinking")
	}
	effort, err := adaptiveEffort(thinking.Effort)
	if err != nil {
		return nil, "", err
	}
	out := &anthropicThinking{Type: "adaptive"}
	if thinking.IncludeOutput {
		out.Display = "summarized"
	}
	return out, effort, nil
}

func adaptiveEffort(effort string) (string, error) {
	value := strings.ToLower(strings.TrimSpace(effort))
	switch value {
	case "":
		return "", nil
	case "minimal":
		return "", fmt.Errorf(`anthropic: thinking effort "minimal" is not supported with adaptive thinking`)
	case "low", "medium", "high", "xhigh", "max":
		return value, nil
	default:
		return "", fmt.Errorf("anthropic: unknown thinking effort %q", effort)
	}
}

func convertTool(tool litellm.Tool) (anthropicTool, error) {
	if tool.Name == "" {
		return anthropicTool{}, fmt.Errorf("anthropic: tool name is required")
	}
	var schema map[string]any
	if len(tool.Parameters) == 0 {
		schema = map[string]any{"type": "object"}
	} else if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
		return anthropicTool{}, fmt.Errorf("anthropic: tool %q parameters must be object schema: %w", tool.Name, err)
	}
	out := anthropicTool{Name: tool.Name, Description: tool.Description, InputSchema: schema}
	switch tool.Strict {
	case litellm.StrictEnabled:
		out.Strict = litellm.Bool(true)
	case litellm.StrictDisabled:
		out.Strict = litellm.Bool(false)
	}
	return out, nil
}

func convertMessages(messages []litellm.Message) (any, []anthropicMessage, error) {
	var system []anthropicContent
	out := make([]anthropicMessage, 0, len(messages))
	for _, msg := range messages {
		content, err := convertBlocks(msg.Blocks)
		if err != nil {
			return nil, nil, err
		}
		switch msg.Role {
		case litellm.RoleSystem:
			system = append(system, content...)
		case litellm.RoleAssistant:
			out = append(out, anthropicMessage{Role: "assistant", Content: content})
		case litellm.RoleUser:
			out = append(out, anthropicMessage{Role: "user", Content: content})
		case litellm.RoleTool:
			out = append(out, anthropicMessage{Role: "user", Content: content})
		default:
			return nil, nil, fmt.Errorf("anthropic: unsupported role %q", msg.Role)
		}
	}
	out = mergeSameRoleMessages(out)
	if err := validateCacheOrder(system, out); err != nil {
		return nil, nil, err
	}
	return systemValue(system), out, nil
}

func validateCacheOrder(system []anthropicContent, messages []anthropicMessage) error {
	seenShort := false
	check := func(block anthropicContent) error {
		if block.CacheControl != nil {
			switch block.CacheControl.TTL {
			case litellm.CacheTTL1h:
				if seenShort {
					return fmt.Errorf("anthropic: 1h cache_control must appear before 5m cache_control")
				}
			case "", litellm.CacheTTL5m:
				seenShort = true
			}
		}
		return nil
	}
	var walk func([]anthropicContent) error
	walk = func(blocks []anthropicContent) error {
		for _, block := range blocks {
			if err := check(block); err != nil {
				return err
			}
			if nested, ok := block.Content.([]anthropicContent); ok {
				if err := walk(nested); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := walk(system); err != nil {
		return err
	}
	for _, msg := range messages {
		if err := walk(msg.Content); err != nil {
			return err
		}
	}
	return nil
}

func systemValue(system []anthropicContent) any {
	if len(system) == 0 {
		return nil
	}
	if len(system) == 1 && system[0].Type == "text" && system[0].CacheControl == nil {
		return system[0].Text
	}
	return system
}

// mergeSameRoleMessages folds consecutive same-role messages into one turn.
// The Messages API requires alternating roles, and parallel tool results must
// all land in a single user message.
func mergeSameRoleMessages(messages []anthropicMessage) []anthropicMessage {
	if len(messages) <= 1 {
		return messages
	}
	out := make([]anthropicMessage, 0, len(messages))
	out = append(out, messages[0])
	for i := 1; i < len(messages); i++ {
		last := &out[len(out)-1]
		if last.Role == messages[i].Role {
			last.Content = append(last.Content, messages[i].Content...)
			continue
		}
		out = append(out, messages[i])
	}
	return out
}

func convertBlocks(blocks []litellm.Block) ([]anthropicContent, error) {
	out := make([]anthropicContent, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContent{Type: "text", Text: b.Text, CacheControl: cache})
		case litellm.ImageBlock:
			source, err := imageSource(b)
			if err != nil {
				return nil, err
			}
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContent{Type: "image", Source: source, CacheControl: cache})
		case litellm.ReasoningBlock:
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			if len(b.Redacted) > 0 {
				out = append(out, anthropicContent{Type: "redacted_thinking", Data: string(b.Redacted), CacheControl: cache})
			} else {
				out = append(out, anthropicContent{Type: "thinking", Thinking: b.Text, Signature: b.Signature, CacheControl: cache})
			}
		case litellm.ToolUseBlock:
			if err := validateToolID(b.ID); err != nil {
				return nil, err
			}
			input := map[string]any{}
			if len(b.Arguments) > 0 {
				if err := json.Unmarshal(b.Arguments, &input); err != nil {
					return nil, fmt.Errorf("anthropic: tool use %q arguments must be object: %w", b.ID, err)
				}
				if input == nil {
					return nil, fmt.Errorf("anthropic: tool use %q arguments must be object", b.ID)
				}
			}
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContent{Type: "tool_use", ID: b.ID, Name: b.Name, Input: &input, CacheControl: cache})
		case litellm.ToolResultBlock:
			if err := validateToolID(b.ToolUseID); err != nil {
				return nil, err
			}
			content, err := convertToolResultContent(b.Content)
			if err != nil {
				return nil, err
			}
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContent{Type: "tool_result", ToolUseID: b.ToolUseID, Content: content, IsError: b.IsError, CacheControl: cache})
		case litellm.ToolReferenceBlock:
			cache, err := cacheControl(b.Cache)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContent{Type: "tool_reference", ToolName: b.ToolName, CacheControl: cache})
		default:
			return nil, fmt.Errorf("anthropic: unsupported block %T", block)
		}
	}
	return out, nil
}

func convertToolResultContent(blocks []litellm.Block) (any, error) {
	if len(blocks) == 1 {
		if text, ok := blocks[0].(litellm.TextBlock); ok && text.Cache == nil {
			return text.Text, nil
		}
	}
	return convertBlocks(blocks)
}

func validateToolID(id string) error {
	if id == "" || strings.IndexFunc(id, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-')
	}) >= 0 {
		return fmt.Errorf("anthropic: tool id %q must contain only letters, digits, underscores or hyphens", id)
	}
	return nil
}
