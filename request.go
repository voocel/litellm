package litellm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"unicode/utf8"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Block interface {
	isBlock()
}

type Annotation struct {
	Type  string
	Text  string
	URL   string
	Extra json.RawMessage
}

type TextBlock struct {
	Text        string
	Annotations []Annotation
	Logprobs    json.RawMessage
	Cache       *CacheControl
}

type ImageBlock struct {
	URL     string
	Data    []byte
	MIME    string
	FileURI string
	Detail  string
	Cache   *CacheControl
}

type ReasoningBlock struct {
	Text      string
	Summary   bool
	Signature string
	Redacted  []byte
	Extra     json.RawMessage
	Cache     *CacheControl
}

type ToolUseBlock struct {
	ID        string
	Name      string
	Arguments json.RawMessage
	Signature string
	Extra     json.RawMessage
	Cache     *CacheControl
}

type ToolResultBlock struct {
	ToolUseID string
	Content   []Block
	IsError   bool
	Cache     *CacheControl
}

type ToolReferenceBlock struct {
	ToolName string
	Extra    json.RawMessage
	Cache    *CacheControl
}

func (TextBlock) isBlock()          {}
func (ImageBlock) isBlock()         {}
func (ReasoningBlock) isBlock()     {}
func (ToolUseBlock) isBlock()       {}
func (ToolResultBlock) isBlock()    {}
func (ToolReferenceBlock) isBlock() {}

type Message struct {
	Role   Role
	Blocks []Block
}

type CacheControl struct {
	Type string
	TTL  string
}

const (
	CacheTypeEphemeral = "ephemeral"
	CacheTTL5m         = "5m"
	CacheTTL1h         = "1h"
)

type Schema json.RawMessage

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

type StrictMode int

const (
	StrictDefault StrictMode = iota
	StrictEnabled
	StrictDisabled
)

type Tool struct {
	Name        string
	Description string
	Parameters  Schema
	Strict      StrictMode
}

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

type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
)

func (c *ToolChoice) Validate() error {
	if c == nil {
		return nil
	}
	if c.Name != "" {
		if c.Mode != "" {
			return NewError(ErrorTypeValidation, "tool choice mode and name are mutually exclusive")
		}
		if !utf8.ValidString(c.Name) {
			return NewError(ErrorTypeValidation, "tool choice name must be valid UTF-8")
		}
		return nil
	}
	switch c.Mode {
	case ToolChoiceAuto, ToolChoiceNone, ToolChoiceRequired:
		return nil
	default:
		return NewError(ErrorTypeValidation, fmt.Sprintf("unsupported tool choice mode %q", c.Mode))
	}
}

type ResponseFormat struct {
	Type       ResponseFormatType
	JSONSchema *JSONSchema
}

type ResponseFormatType string

const (
	ResponseFormatText       ResponseFormatType = "text"
	ResponseFormatJSONObject ResponseFormatType = "json_object"
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

type JSONSchema struct {
	Name        string
	Description string
	Schema      Schema
	Strict      StrictMode
}

type Thinking struct {
	Mode          ThinkingMode
	Effort        string
	BudgetTokens  *int
	IncludeOutput bool
}

func (t *Thinking) HasOptions() bool {
	return t != nil && (t.Effort != "" || t.BudgetTokens != nil || t.IncludeOutput)
}

func (t *Thinking) Validate() error {
	if t == nil {
		return nil
	}
	if !utf8.ValidString(t.Effort) {
		return NewError(ErrorTypeValidation, "thinking effort must be valid UTF-8")
	}
	if t.Mode == ThinkingUnspecified && t.HasOptions() {
		return NewError(ErrorTypeValidation, "thinking mode must be enabled or disabled when thinking options are set")
	}
	if t.Mode == ThinkingDisabled && t.HasOptions() {
		return NewError(ErrorTypeValidation, "thinking options cannot be set when thinking is disabled")
	}
	return nil
}

type ThinkingMode int

const (
	ThinkingUnspecified ThinkingMode = iota
	ThinkingDisabled
	ThinkingEnabled
)

type CachePolicy struct {
	Retention string
	Placement CachePlacement
}

type CachePlacement string

const (
	CachePlacementPrefix CachePlacement = "prefix"
)

// ProviderOptions contains JSON values owned by the request. Use NewProviderOptions
// or Set to encode Go values; the client copies each value before observation or execution.
type ProviderOptions map[string]json.RawMessage

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
	Cache          *CachePolicy

	ProviderOptions ProviderOptions

	captureRawResponse bool
}

func (r *Request) CaptureRawResponse() bool {
	return r != nil && r.captureRawResponse
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
		return NewError(ErrorTypeValidation, "provider options map is nil")
	}
	if !utf8.ValidString(key) {
		return NewError(ErrorTypeValidation, "provider option key must be valid UTF-8")
	}
	data, err := json.Marshal(value)
	if err != nil {
		return NewError(ErrorTypeValidation, fmt.Sprintf("provider option %q: %v", key, err))
	}
	o[key] = data
	return nil
}

// Validate checks raw JSON at the request boundary, including values inserted
// directly rather than through Set.
func (o ProviderOptions) Validate() error {
	for key, value := range o {
		if !utf8.ValidString(key) {
			return NewError(ErrorTypeValidation, "provider option key must be valid UTF-8")
		}
		if !utf8.Valid(value) || !json.Valid(value) {
			return NewError(ErrorTypeValidation, fmt.Sprintf("provider option %q must be valid UTF-8 JSON", key))
		}
	}
	return nil
}

// Decode gives a provider an independent JSON tree. Numbers remain json.Number
// to preserve integer precision until the provider validates its wire type.
func (o ProviderOptions) Decode() (map[string]any, error) {
	if err := o.Validate(); err != nil {
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
			return nil, NewError(ErrorTypeValidation, fmt.Sprintf("provider option %q: %v", key, err))
		}
		values[key] = value
	}
	return values, nil
}
