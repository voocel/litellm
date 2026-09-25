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
	Type  string
	Text  string
	URL   string
	Extra json.RawMessage
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
	Provider string
	// Model is the requested model.
	Model string
	// Data holds the vendor's native fields as JSON.
	Data json.RawMessage
}

// TextBlock is plain text. Annotations and Logprobs are response metadata.
type TextBlock struct {
	Text        string
	Annotations []Annotation
	Logprobs    json.RawMessage
	State       *ProviderState
	Cache       *CacheControl
}

// ImageBlock is an image. Set one source: URL, Data with MIME, or FileURI, a
// vendor file reference. Detail is sent where the wire format has it.
type ImageBlock struct {
	URL     string
	Data    []byte
	MIME    string
	FileURI string
	Detail  string
	Cache   *CacheControl
}

// ReasoningBlock is model reasoning. Summary marks Text as a summary rather
// than the full reasoning. Text is empty when the vendor returns the reasoning
// encrypted or redacted; State then carries it.
type ReasoningBlock struct {
	Text    string
	Summary bool
	State   *ProviderState
	Cache   *CacheControl
}

// ToolUseBlock is a tool call from the assistant. Arguments is the JSON the
// model produced.
type ToolUseBlock struct {
	ID        string
	Name      string
	Arguments json.RawMessage
	State     *ProviderState
	Cache     *CacheControl
}

// ToolResultBlock answers the ToolUseBlock with ID ToolUseID. Content holds
// TextBlock, ImageBlock or ToolReferenceBlock values.
type ToolResultBlock struct {
	ToolUseID string
	Content   []Block
	IsError   bool
	Cache     *CacheControl
}

// ToolReferenceBlock names a tool inside tool result content, as returned by a
// tool search tool.
type ToolReferenceBlock struct {
	ToolName string
	Cache    *CacheControl
}

func (TextBlock) isBlock()          {}
func (ImageBlock) isBlock()         {}
func (ReasoningBlock) isBlock()     {}
func (ToolUseBlock) isBlock()       {}
func (ToolResultBlock) isBlock()    {}
func (ToolReferenceBlock) isBlock() {}

// Message is one conversation turn.
type Message struct {
	Role   Role
	Blocks []Block
}

// CacheControl marks a cache breakpoint: the prompt prefix up to and including
// this block may be cached. TTL is passed to the vendor as is; empty selects
// the vendor default.
type CacheControl struct {
	TTL string
}

// Common CacheControl TTL values.
const (
	CacheTTL5m = "5m"
	CacheTTL1h = "1h"
)

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

// StrictMode controls strict schema adherence. StrictDefault leaves the wire
// flag unset.
type StrictMode int

const (
	StrictDefault StrictMode = iota
	StrictEnabled
	StrictDisabled
)

// Value returns the wire strict flag and whether the mode sets one.
func (m StrictMode) Value() (strict, set bool) {
	return m == StrictEnabled, m != StrictDefault
}

// Tool declares a function the model may call.
type Tool struct {
	Name        string
	Description string
	Parameters  Schema
	Strict      StrictMode
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
	Mode ToolChoiceMode
	Name string
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
// ResponseFormatJSONSchema.
type ResponseFormat struct {
	Type       ResponseFormatType
	JSONSchema *JSONSchema
}

// ResponseFormatType selects the output format.
type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// JSONSchema is a named schema for structured output.
type JSONSchema struct {
	Name        string
	Description string
	Schema      Schema
	Strict      StrictMode
}

// Thinking configures model reasoning. Effort and BudgetTokens are sent as
// given; which values a model accepts is the vendor's decision. IncludeOutput
// is a hint to return reasoning where the vendor makes it optional; providers
// without such a switch ignore it.
type Thinking struct {
	Mode          ThinkingMode
	Effort        string
	BudgetTokens  *int
	IncludeOutput bool
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
	switch t.Mode {
	case ThinkingEnabled:
	case ThinkingDisabled:
		if t.hasOptions() {
			return NewError("", ErrorTypeValidation, "thinking options cannot be set when thinking is disabled", nil)
		}
	default:
		return NewError("", ErrorTypeValidation, fmt.Sprintf("unknown thinking mode %d", t.Mode), nil)
	}
	if t.BudgetTokens != nil && *t.BudgetTokens <= 0 {
		return NewError("", ErrorTypeValidation, "thinking budget_tokens must be positive", nil)
	}
	return nil
}

// ThinkingMode is ThinkingEnabled by default; a nil *Thinking leaves thinking
// to the vendor default.
type ThinkingMode int

const (
	ThinkingEnabled ThinkingMode = iota
	ThinkingDisabled
)

// ProviderOptions contains JSON values owned by the request. Use NewProviderOptions
// or Set to encode Go values; the client copies each value before observation or execution.
type ProviderOptions map[string]json.RawMessage

// Request is a provider-neutral chat request. Nil pointers and empty fields
// are omitted from the wire, leaving the vendor default.
type Request struct {
	Model    string
	Messages []Message

	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	Stop        []string

	Tools      []Tool
	ToolChoice *ToolChoice

	ResponseFormat *ResponseFormat
	Thinking       *Thinking

	ProviderOptions ProviderOptions
}

func cloneBytes(b []byte) []byte {
	if len(b) == 0 {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
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
