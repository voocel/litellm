package openaicompat

import (
	"context"
	"errors"

	"github.com/voocel/litellm"
)

// Config holds the connection settings shared by compat and the vendor
// wrappers.
type Config struct {
	// Name, when set, names the provider instead of the vendor, as a
	// compatible endpoint should: provider state and errors carry it.
	Name string
	// APIKey is sent as a bearer token; APIKeyFunc, when set, resolves it per
	// request instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// BaseURL is the API root; wrappers default it to the vendor endpoint.
	BaseURL string
	// HTTPClient sends requests; nil uses http.DefaultClient. Wrap it with
	// retry.NewHTTPClient to retry.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are set after the defaults, so they can override them.
	Headers map[string]string

	// AllowUnknownProviderOptions copies ProviderOptions the provider does not
	// list into the JSON body as is. The default rejects them.
	AllowUnknownProviderOptions bool
}

// Spec describes one vendor's dialect of the Chat Completions protocol. Hooks
// map structure only; they never validate values or inspect the model name.
type Spec struct {
	Name           string
	BaseURL        string // used when Config.BaseURL is empty
	APIKeyRequired bool
	// ModelsPath lists the models; empty means /models. It must take the
	// API key, so that listing checks it.
	ModelsPath string

	// MaxTokensField names the output limit field; empty means max_tokens.
	MaxTokensField string
	// Thinking maps Request.Thinking to body fields. Nil sends Effort as
	// reasoning_effort and Thinking.Disabled as reasoning_effort "none".
	Thinking func(*litellm.Thinking) (map[string]any, error)
	// ThinkingAlwaysOn reports that the vendor cannot disable thinking, so
	// Thinking.Disabled is rejected.
	ThinkingAlwaysOn bool
	// Fields are sent on every request.
	Fields map[string]any
	// Options lists the accepted ProviderOptions keys. Options are copied into
	// the body; one naming a generated object field is merged into it.
	Options []string
	// Cache holds the content part fields of a block cache breakpoint. Nil
	// drops breakpoints, which are hints.
	Cache map[string]any
	// ReasoningFields names the message fields that carry reasoning, in
	// priority order. reasoning_details is kept as the ReasoningBlock State and
	// replayed verbatim to this provider; history text is sent in the first
	// other field.
	ReasoningFields []string
	// StringContentRoles requires string content for these roles. Text blocks
	// are concatenated without separators; images are rejected.
	StringContentRoles []litellm.Role
	// ImageFileID encodes ImageBlock.FileURI as a file part with a file_id,
	// instead of using it as image_url.url.
	ImageFileID bool
	// SchemaFallback moves JSON Schema into a prompt and uses this wire format:
	// json_object enables JSON mode; text uses prompting alone. Empty keeps the
	// native JSON Schema format. Neither fallback enforces schema adherence.
	SchemaFallback litellm.ResponseFormatType
	// OmitStreamOptions leaves stream_options out of stream requests.
	OmitStreamOptions bool
}

func (s Spec) maxTokensField() string {
	if s.MaxTokensField != "" {
		return s.MaxTokensField
	}
	return "max_tokens"
}

// reasoningEffort is the Chat Completions thinking mapping used when
// Spec.Thinking is nil.
func reasoningEffort(thinking *litellm.Thinking) (map[string]any, error) {
	if thinking.BudgetTokens != nil {
		return nil, errors.New("thinking budget_tokens is not supported; use effort")
	}
	if thinking.Disabled {
		return map[string]any{"reasoning_effort": "none"}, nil
	}
	if thinking.Effort != "" {
		return map[string]any{"reasoning_effort": thinking.Effort}, nil
	}
	return nil, nil
}

// ThinkingType maps Thinking to {"thinking": {"type": on | "disabled"}}, the
// switch several vendors use. With effort, Effort is sent as reasoning_effort;
// without, it is rejected. BudgetTokens is always rejected.
func ThinkingType(on string, effort bool) func(*litellm.Thinking) (map[string]any, error) {
	return func(thinking *litellm.Thinking) (map[string]any, error) {
		if thinking.BudgetTokens != nil {
			return nil, errors.New("thinking budget_tokens is not supported")
		}
		if thinking.Disabled {
			return map[string]any{"thinking": map[string]any{"type": "disabled"}}, nil
		}
		body := map[string]any{"thinking": map[string]any{"type": on}}
		if thinking.Effort != "" {
			if !effort {
				return nil, errors.New("thinking effort is not supported")
			}
			body["reasoning_effort"] = thinking.Effort
		}
		return body, nil
	}
}
