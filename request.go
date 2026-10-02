package litellm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"unicode/utf8"
)

// Role identifies the author of a Message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Block is one piece of message content. The implementations are TextBlock,
// ImageBlock, ReasoningBlock, ToolUseBlock, ToolResultBlock and
// ToolReferenceBlock.
type Block interface {
	isBlock()
}

// Annotation is a citation attached to text. Extra keeps the vendor entry
// verbatim.
type Annotation struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	URL   string          `json:"url,omitempty"`
	Extra json.RawMessage `json:"extra,omitempty"`
}

// ProviderState is data a provider attaches to a block it produced so the
// block can be sent back to it: a reasoning signature, encrypted reasoning or
// an item id. Keep it with the block when storing history; only the provider
// named by Provider reads it.
//
// Replay rule: portable content (text, reasoning text, tool calls) maps to
// wherever the target wire format can carry it, while State is sent only to
// the provider that produced it. A block the wire cannot carry without its
// State, such as Anthropic thinking, is dropped; where the vendor documents a
// placeholder for foreign content, the provider supplies it. Blocks built by
// the caller (nil State) count as foreign.
type ProviderState struct {
	// Provider is the Name of the provider that produced the block.
	Provider string `json:"provider"`
	// Model is the requested model.
	Model string `json:"model,omitempty"`
	// Data holds the vendor's native fields as JSON.
	Data json.RawMessage `json:"data,omitempty"`
}

// TextBlock is plain text. Annotations and Logprobs are response metadata.
type TextBlock struct {
	Text        string          `json:"text"`
	Annotations []Annotation    `json:"annotations,omitempty"`
	Logprobs    json.RawMessage `json:"logprobs,omitempty"`
	State       *ProviderState  `json:"state,omitempty"`
	Cache       *CacheControl   `json:"cache,omitempty"`
}

// ImageBlock is an image. Set one source: URL, Data with MIME, or FileURI, a
// vendor file reference. Detail is sent where the wire format has it.
type ImageBlock struct {
	URL     string        `json:"url,omitempty"`
	Data    []byte        `json:"data,omitempty"`
	MIME    string        `json:"mime,omitempty"`
	FileURI string        `json:"file_uri,omitempty"`
	Detail  string        `json:"detail,omitempty"`
	Cache   *CacheControl `json:"cache,omitempty"`
}

// ReasoningBlock is model reasoning. Summary marks Text as a summary rather
// than the full reasoning. Text is empty when the vendor returns the reasoning
// encrypted or redacted; State then carries it.
type ReasoningBlock struct {
	Text    string         `json:"text"`
	Summary bool           `json:"summary,omitempty"`
	State   *ProviderState `json:"state,omitempty"`
}

// ToolUseBlock is a tool call from the assistant. Arguments is the text the
// model produced, which is meant to be a JSON object but may not be one, as
// when the response was cut off at the output limit.
type ToolUseBlock struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments string         `json:"arguments,omitempty"`
	State     *ProviderState `json:"state,omitempty"`
	Cache     *CacheControl  `json:"cache,omitempty"`
}

// ToolResultBlock answers the ToolUseBlock with ID ToolUseID. Content holds
// TextBlock, ImageBlock or ToolReferenceBlock values.
type ToolResultBlock struct {
	ToolUseID string        `json:"tool_use_id"`
	Content   []Block       `json:"content,omitempty"`
	IsError   bool          `json:"is_error,omitempty"`
	Cache     *CacheControl `json:"cache,omitempty"`
}

// ToolReferenceBlock names a tool inside tool result content, as returned by a
// tool search tool.
type ToolReferenceBlock struct {
	ToolName string        `json:"tool_name"`
	Cache    *CacheControl `json:"cache,omitempty"`
}

func (TextBlock) isBlock()          {}
func (ImageBlock) isBlock()         {}
func (ReasoningBlock) isBlock()     {}
func (ToolUseBlock) isBlock()       {}
func (ToolResultBlock) isBlock()    {}
func (ToolReferenceBlock) isBlock() {}

// Message is one conversation turn.
type Message struct {
	Role   Role    `json:"role"`
	Blocks []Block `json:"blocks"`
}

// CacheControl marks a cache breakpoint: the prompt prefix up to and including
// this block may be cached, for the vendor's default time.
type CacheControl struct{}

// Schema is a JSON Schema document.
type Schema json.RawMessage

// SchemaFrom returns v as a Schema. JSON text (Schema, json.RawMessage, []byte
// or string) is validated and copied; any other value is marshaled. A nil v
// returns a nil Schema.
func SchemaFrom(v any) (Schema, error) {
	switch s := v.(type) {
	case nil:
		return nil, nil
	case Schema:
		return cloneBytes([]byte(s)), nil
	case json.RawMessage:
		if !json.Valid(s) {
			return nil, fmt.Errorf("schema must be valid JSON")
		}
		return Schema(cloneBytes(s)), nil
	case []byte:
		if !json.Valid(s) {
			return nil, fmt.Errorf("schema must be valid JSON")
		}
		return Schema(cloneBytes(s)), nil
	case string:
		b := []byte(s)
		if !json.Valid(b) {
			return nil, fmt.Errorf("schema must be valid JSON")
		}
		return Schema(cloneBytes(b)), nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("marshal schema: %w", err)
		}
		if !json.Valid(b) {
			return nil, fmt.Errorf("schema must be valid JSON")
		}
		return Schema(b), nil
	}
}

// Tool declares a function the model may call. Strict, when set, asks the
// vendor to enforce, or not, that calls fit Parameters; nil leaves its
// default.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  Schema `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// NewTool builds a Tool, converting parameters with SchemaFrom.
func NewTool(name, description string, parameters any) (Tool, error) {
	schema, err := SchemaFrom(parameters)
	if err != nil {
		return Tool{}, err
	}
	return Tool{Name: name, Description: description, Parameters: schema}, nil
}

// ToolChoice selects a policy, or a named tool when Name is set. A nil choice
// leaves selection to the provider. Mode and Name are mutually exclusive.
type ToolChoice struct {
	Mode ToolChoiceMode `json:"mode,omitempty"`
	Name string         `json:"name,omitempty"`
}

// ToolChoiceMode is a tool selection policy.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
)

// validate reports whether c is well formed. A nil choice is valid.
func (c *ToolChoice) validate() error {
	if c == nil {
		return nil
	}
	if c.Name != "" {
		if c.Mode != "" {
			return NewError("", ErrorTypeValidation, "tool choice mode and name are mutually exclusive", nil)
		}
		if !utf8.ValidString(c.Name) {
			return NewError("", ErrorTypeValidation, "tool choice name must be valid UTF-8", nil)
		}
		return nil
	}
	switch c.Mode {
	case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired:
		return nil
	default:
		return NewError("", ErrorTypeValidation, fmt.Sprintf("unsupported tool choice mode %q", c.Mode), nil)
	}
}

// ResponseFormat constrains the output format. JSONSchema is used with
// ResponseFormatJSONSchema. Providers without native schema support may use a
// prompt instead and report a Warning; this does not enforce schema adherence,
// including when Strict is set. See providers.md for the mapping.
type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema *JSONSchema        `json:"json_schema,omitempty"`
}

// ResponseFormatType selects the output format.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// JSONSchema is a named schema for structured output. Strict is as in Tool.
type JSONSchema struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Schema      Schema `json:"schema,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// Thinking configures model reasoning; a nil *Thinking leaves it to the
// vendor default. Disabled turns reasoning off and allows no other field.
// Effort and BudgetTokens are sent as given; which values a model accepts is
// the vendor's decision. IncludeOutput is a hint to return reasoning where
// the vendor makes it optional; providers without such a switch ignore it.
type Thinking struct {
	Disabled      bool   `json:"disabled,omitempty"`
	Effort        string `json:"effort,omitempty"`
	BudgetTokens  *int   `json:"budget_tokens,omitempty"`
	IncludeOutput bool   `json:"include_output,omitempty"`
}

func (t *Thinking) hasOptions() bool {
	return t != nil && (t.Effort != "" || t.BudgetTokens != nil || t.IncludeOutput)
}

// validate checks the combination only; vendor values are not checked.
// A nil Thinking is valid.
func (t *Thinking) validate() error {
	if t == nil {
		return nil
	}
	if !utf8.ValidString(t.Effort) {
		return NewError("", ErrorTypeValidation, "thinking effort must be valid UTF-8", nil)
	}
	if t.Disabled && t.hasOptions() {
		return NewError("", ErrorTypeValidation, "thinking options cannot be set when thinking is disabled", nil)
	}
	if t.BudgetTokens != nil && *t.BudgetTokens <= 0 {
		return NewError("", ErrorTypeValidation, "thinking budget_tokens must be positive", nil)
	}
	return nil
}

// ProviderOptions contains JSON values owned by the request. Use NewProviderOptions
// or Set to encode Go values; the client copies each value before observation or execution.
type ProviderOptions map[string]json.RawMessage

// Request is a provider-neutral chat request. Nil pointers and empty fields
// are omitted from the wire, leaving the vendor default.
type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`

	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`

	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	Thinking       *Thinking       `json:"thinking,omitempty"`

	ProviderOptions ProviderOptions `json:"provider_options,omitempty"`
}

func cloneBytes(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
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
		b.State = cloneState(b.State)
		return b
	case ImageBlock:
		b.Data = cloneBytes(b.Data)
		return b
	case ReasoningBlock:
		b.State = cloneState(b.State)
		return b
	case ToolUseBlock:
		b.State = cloneState(b.State)
		return b
	case ToolResultBlock:
		b.Content = cloneBlocks(b.Content)
		return b
	default:
		return block
	}
}

func cloneTools(tools []Tool) []Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]Tool, len(tools))
	for i, tool := range tools {
		out[i] = tool
		out[i].Parameters = Schema(cloneBytes(tool.Parameters))
		out[i].Strict = clonePtr(tool.Strict)
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
		schema.Strict = clonePtr(format.JSONSchema.Strict)
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

func cloneState(state *ProviderState) *ProviderState {
	if state == nil {
		return nil
	}
	out := *state
	out.Data = cloneBytes(state.Data)
	return &out
}

// NewProviderOptions serializes values immediately, so subsequent mutations of
// the supplied Go objects cannot change the request. Encoding errors are returned.
func NewProviderOptions(values map[string]any) (ProviderOptions, error) {
	options := make(ProviderOptions, len(values))
	for key, value := range values {
		if err := options.Set(key, value); err != nil {
			return nil, err
		}
	}
	return options, nil
}

// Set encodes a value into an initialized options map.
func (o ProviderOptions) Set(key string, value any) error {
	if o == nil {
		return NewError("", ErrorTypeValidation, "provider options map is nil", nil)
	}
	if !utf8.ValidString(key) {
		return NewError("", ErrorTypeValidation, "provider option key must be valid UTF-8", nil)
	}
	data, err := json.Marshal(value)
	if err != nil {
		return NewError("", ErrorTypeValidation, fmt.Sprintf("provider option %q: %v", key, err), err)
	}
	o[key] = data
	return nil
}

// validate checks raw JSON at the request boundary, including values inserted
// directly rather than through Set.
func (o ProviderOptions) validate() error {
	for key, value := range o {
		if !utf8.ValidString(key) {
			return NewError("", ErrorTypeValidation, "provider option key must be valid UTF-8", nil)
		}
		if !utf8.Valid(value) || !json.Valid(value) {
			return NewError("", ErrorTypeValidation, fmt.Sprintf("provider option %q must be valid UTF-8 JSON", key), nil)
		}
	}
	return nil
}

// Decode gives a provider an independent JSON tree. Numbers remain json.Number
// to preserve integer precision until the provider validates its wire type.
func (o ProviderOptions) Decode() (map[string]any, error) {
	if err := o.validate(); err != nil {
		return nil, err
	}
	if o == nil {
		return nil, nil
	}
	values := make(map[string]any, len(o))
	for key, raw := range o {
		var value any
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if err := decoder.Decode(&value); err != nil {
			return nil, NewError("", ErrorTypeValidation, fmt.Sprintf("provider option %q: %v", key, err), err)
		}
		values[key] = value
	}
	return values, nil
}
