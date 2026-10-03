package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/claude"
	"github.com/voocel/litellm/internal/wire"
)

// ProviderOptions are native Messages API fields copied into the body. An
// option naming a generated object is merged into it; setting a field the
// request already generated is an error. Options whose results litellm does
// not model, such as server tools, are not offered.
const (
	ProviderOptionMetadata     = "metadata"
	ProviderOptionServiceTier  = "service_tier"
	ProviderOptionTopK         = "top_k"
	ProviderOptionOutputConfig = "output_config"
	// ProviderOptionThinking sends a thinking object as given when
	// Request.Thinking is nil, for shapes litellm does not map, and is merged
	// into the generated one otherwise, e.g. {"display": "omitted"}.
	ProviderOptionThinking = "thinking"
	// ProviderOptionToolChoice is merged into the generated tool_choice, e.g.
	// {"disable_parallel_tool_use": true}.
	ProviderOptionToolChoice = "tool_choice"
)

var providerOptions = []string{
	ProviderOptionMetadata, ProviderOptionServiceTier, ProviderOptionTopK, ProviderOptionOutputConfig,
	ProviderOptionThinking, ProviderOptionToolChoice,
}

func sortedOptions() []string {
	out := slices.Clone(providerOptions)
	slices.Sort(out)
	return out
}

type request struct {
	Model         string                 `json:"model"`
	System        any                    `json:"system,omitempty"`
	MaxTokens     int                    `json:"max_tokens"`
	Messages      []message              `json:"messages"`
	Stream        bool                   `json:"stream,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	Tools         []tool                 `json:"tools,omitempty"`
	ToolChoice    map[string]any         `json:"tool_choice,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Thinking      *claude.ThinkingConfig `json:"thinking,omitempty"`
	OutputConfig  map[string]any         `json:"output_config,omitempty"`
}

type message struct {
	Role    string    `json:"role"`
	Content []content `json:"content"`
}

type content struct {
	Type         string          `json:"type"`
	Text         string          `json:"text,omitempty"`
	Source       *imageSource    `json:"source,omitempty"`
	Thinking     *string         `json:"thinking,omitempty"` // set on thinking blocks, which require it even when empty
	Signature    string          `json:"signature,omitempty"`
	Data         string          `json:"data,omitempty"`
	ID           string          `json:"id,omitempty"`
	ToolUseID    string          `json:"tool_use_id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        json.RawMessage `json:"input,omitempty"`
	Content      any             `json:"content,omitempty"`
	ToolName     string          `json:"tool_name,omitempty"`
	IsError      bool            `json:"is_error,omitempty"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
	// Citations is read from responses only.
	Citations []json.RawMessage `json:"citations,omitempty"`
}

// thinkingState is the ProviderState of a thinking or redacted_thinking block:
// the fields besides the text needed to send it back.
type thinkingState struct {
	Type      string `json:"type"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type cacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      *bool           `json:"strict,omitempty"`
}

func buildRequest(req *litellm.Request, provider string, stream bool) ([]byte, error) {
	if req.MaxTokens == nil {
		return nil, errors.New("max_tokens is required by the Messages API")
	}
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if err := wire.CheckOptions(opts, providerOptions); err != nil {
		return nil, err
	}
	out := &request{
		Model:         req.Model,
		MaxTokens:     *req.MaxTokens,
		Stream:        stream,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.Stop,
		Thinking:      claude.Thinking(req.Thinking),
	}
	if choice := req.ToolChoice; choice != nil {
		out.ToolChoice = convertToolChoice(choice)
	}
	if out.Thinking != nil && out.Thinking.Effort != "" {
		out.OutputConfig = map[string]any{"effort": out.Thinking.Effort}
	}
	if format, err := convertResponseFormat(req.ResponseFormat); err != nil {
		return nil, err
	} else if format != nil {
		if out.OutputConfig == nil {
			out.OutputConfig = map[string]any{}
		}
		out.OutputConfig["format"] = format
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, convertTool(t))
	}
	if out.System, out.Messages, err = convertMessages(req.Messages, provider); err != nil {
		return nil, err
	}
	return wire.MarshalBody(out, opts)
}

func convertToolChoice(choice *litellm.ToolChoice) map[string]any {
	if choice.Name != "" {
		return map[string]any{"type": "tool", "name": choice.Name}
	}
	if choice.Mode == litellm.ToolChoiceRequired {
		return map[string]any{"type": "any"}
	}
	return map[string]any{"type": string(choice.Mode)}
}

func convertResponseFormat(format *litellm.ResponseFormat) (map[string]any, error) {
	if format == nil {
		return nil, nil
	}
	switch format.Type {
	case "", litellm.ResponseFormatText:
		return nil, nil
	case litellm.ResponseFormatJSONSchema:
		out := map[string]any{"type": "json_schema"}
		if len(format.JSONSchema.Schema) > 0 {
			out["schema"] = json.RawMessage(format.JSONSchema.Schema)
		}
		return out, nil
	case litellm.ResponseFormatJSONObject:
		return nil, errors.New("response_format json_object has no Messages API equivalent; use json_schema")
	default:
		return nil, fmt.Errorf("unsupported response format %q", format.Type)
	}
}

func convertTool(t litellm.Tool) tool {
	out := tool{Name: t.Name, Description: t.Description, InputSchema: json.RawMessage(`{"type":"object"}`)}
	if len(t.Parameters) > 0 {
		out.InputSchema = json.RawMessage(t.Parameters)
	}
	out.Strict = t.Strict
	return out
}

// convertMessages sends leading system messages as the system field and later
// ones in place, where changing them keeps the cached prefix and thinking
// valid. Messages left empty, such as one holding only foreign reasoning, are
// omitted.
func convertMessages(messages []litellm.Message, provider string) (any, []message, error) {
	var system []content
	out := make([]message, 0, len(messages))
	for i, msg := range messages {
		blocks, err := convertBlocks(msg.Blocks, provider)
		if err != nil {
			return nil, nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		if len(blocks) == 0 {
			continue
		}
		role := "user"
		switch msg.Role {
		case litellm.RoleSystem:
			if len(out) == 0 {
				system = append(system, blocks...)
				continue
			}
			role = "system"
		case litellm.RoleAssistant:
			role = "assistant"
		}
		// Roles must alternate, and parallel tool results share one user turn.
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			continue
		}
		out = append(out, message{Role: role, Content: blocks})
	}
	if len(system) == 1 && system[0].Type == "text" && system[0].CacheControl == nil {
		return system[0].Text, out, nil
	}
	if len(system) == 0 {
		return nil, out, nil
	}
	return system, out, nil
}

func convertBlocks(blocks []litellm.Block, provider string) ([]content, error) {
	out := make([]content, 0, len(blocks))
	for _, block := range blocks {
		var c content
		switch b := block.(type) {
		case litellm.TextBlock:
			if b.Text == "" {
				continue // empty text blocks are rejected, e.g. Gemini's signature-only parts
			}
			c = content{Type: "text", Text: b.Text, CacheControl: convertCache(b.Cache)}
		case litellm.ImageBlock:
			source, err := convertImage(b)
			if err != nil {
				return nil, err
			}
			c = content{Type: "image", Source: source, CacheControl: convertCache(b.Cache)}
		case litellm.ReasoningBlock:
			// Thinking is valid only with the signature Claude issued, so
			// reasoning from elsewhere is dropped.
			state, ok := wire.ReadState[thinkingState](b.State, provider)
			if !ok {
				continue
			}
			c = content{Type: state.Type, Signature: state.Signature, Data: state.Data}
			if state.Type == "thinking" {
				c.Thinking = new(b.Text)
			}
		case litellm.ToolUseBlock:
			input, err := toolInput(b)
			if err != nil {
				return nil, err
			}
			c = content{Type: "tool_use", ID: claude.ToolUseID(b.ID), Name: b.Name, Input: input, CacheControl: convertCache(b.Cache)}
		case litellm.ToolResultBlock:
			result, err := convertToolResult(b.Content, provider)
			if err != nil {
				return nil, err
			}
			c = content{Type: "tool_result", ToolUseID: claude.ToolUseID(b.ToolUseID), Content: result, IsError: b.IsError, CacheControl: convertCache(b.Cache)}
		case litellm.ToolReferenceBlock:
			c = content{Type: "tool_reference", ToolName: b.ToolName, CacheControl: convertCache(b.Cache)}
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
		out = append(out, c)
	}
	return out, nil
}

// toolInput returns the arguments as the input object the protocol requires.
func toolInput(b litellm.ToolUseBlock) (json.RawMessage, error) {
	if b.Arguments == "" {
		return json.RawMessage("{}"), nil
	}
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(b.Arguments), &object) != nil || object == nil {
		return nil, fmt.Errorf("tool use %q (%s) arguments are not a JSON object", b.ID, b.Name)
	}
	return json.RawMessage(b.Arguments), nil
}

func convertToolResult(blocks []litellm.Block, provider string) (any, error) {
	if len(blocks) == 0 {
		return nil, nil
	}
	if len(blocks) == 1 {
		if text, ok := blocks[0].(litellm.TextBlock); ok && text.Cache == nil {
			return text.Text, nil
		}
	}
	return convertBlocks(blocks, provider)
}

func convertCache(cache *litellm.CacheControl) *cacheControl {
	if cache == nil {
		return nil
	}
	return &cacheControl{Type: "ephemeral", TTL: cache.TTL}
}

func convertImage(block litellm.ImageBlock) (*imageSource, error) {
	switch {
	case block.URL != "":
		// The url source takes hosted images only.
		if mime, data, ok := wire.ParseDataURL(block.URL); ok {
			return &imageSource{Type: "base64", MediaType: mime, Data: data}, nil
		}
		return &imageSource{Type: "url", URL: block.URL}, nil
	case len(block.Data) > 0:
		if block.MIME == "" {
			return nil, errors.New("inline image requires MIME")
		}
		return &imageSource{Type: "base64", MediaType: block.MIME, Data: base64.StdEncoding.EncodeToString(block.Data)}, nil
	case block.FileURI != "":
		return nil, errors.New("image FileURI is not supported")
	default:
		return nil, errors.New("image requires URL or data")
	}
}
