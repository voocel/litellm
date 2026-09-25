package gemini

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// ProviderOptions are native request fields copied into the body. An option
// naming a generated object or array is merged into or appended to it, e.g.
// {"topK": 40} under "generationConfig" or [{"googleSearch": {}}] under
// "tools".
const (
	ProviderOptionSafetySettings   = "safetySettings"
	ProviderOptionGenerationConfig = "generationConfig"
	ProviderOptionTools            = "tools"
	ProviderOptionToolConfig       = "toolConfig"
	ProviderOptionCachedContent    = "cachedContent"
)

var providerOptions = []string{
	ProviderOptionSafetySettings, ProviderOptionGenerationConfig, ProviderOptionTools,
	ProviderOptionToolConfig, ProviderOptionCachedContent,
}

func sortedOptions() []string {
	out := slices.Clone(providerOptions)
	slices.Sort(out)
	return out
}

func buildRequest(req *litellm.Request) ([]byte, error) {
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if err := wire.CheckOptions(opts, providerOptions); err != nil {
		return nil, err
	}
	out := &request{}
	contents, system, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	out.Contents = contents
	if len(system) > 0 {
		out.SystemInstruction = &content{Parts: system}
	}
	if out.GenerationConfig, err = convertGenerationConfig(req); err != nil {
		return nil, err
	}
	if len(req.Tools) > 0 {
		declarations, strict, err := convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.Tools = []tool{{FunctionDeclarations: declarations}}
		out.ToolConfig = convertToolChoice(req.ToolChoice, strict)
	}
	return wire.MarshalBody(out, opts)
}

func convertMessages(messages []litellm.Message) ([]content, []part, error) {
	out := make([]content, 0, len(messages))
	var system []part
	// Function responses name their call; the name comes from the tool use.
	names := make(map[string]string)
	for i, msg := range messages {
		var parts []part
		var err error
		role := "user"
		switch msg.Role {
		case litellm.RoleSystem:
			parts, err = convertBlocks(msg.Blocks, names)
			system = append(system, parts...)
			if err != nil {
				return nil, nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			continue
		case litellm.RoleAssistant:
			role = "model"
		}
		if parts, err = convertBlocks(msg.Blocks, names); err != nil {
			return nil, nil, fmt.Errorf("messages[%d]: %w", i, err)
		}
		if len(parts) == 0 {
			continue
		}
		// Responses to parallel calls must share one turn.
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Parts = append(out[n-1].Parts, parts...)
			continue
		}
		out = append(out, content{Role: role, Parts: parts})
	}
	for _, c := range out {
		if c.Role == "model" {
			signFirstCall(c.Parts)
		}
	}
	return out, system, nil
}

// skipSignature is the placeholder Gemini documents for function calls it did
// not sign. It is sent as this literal string, which the Gemini API and
// Vertex AI accept, not base64-encoded like real signatures.
const skipSignature = "skip_thought_signature_validator"

// signFirstCall gives the first function call of a model turn the placeholder
// signature when it has none, such as one from another provider: that call
// must carry a signature.
func signFirstCall(parts []part) {
	for i := range parts {
		if parts[i].FunctionCall != nil {
			if parts[i].ThoughtSignature == "" {
				parts[i].ThoughtSignature = skipSignature
			}
			return
		}
	}
}

func convertBlocks(blocks []litellm.Block, names map[string]string) ([]part, error) {
	out := make([]part, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if sig := signature(b.State); b.Text != "" || sig != "" {
				out = append(out, part{Text: new(b.Text), ThoughtSignature: sig})
			}
		case litellm.ImageBlock:
			converted, err := convertImage(b)
			if err != nil {
				return nil, err
			}
			out = append(out, converted)
		case litellm.ReasoningBlock:
			// Reasoning from elsewhere is sent as an unsigned thought.
			if sig := signature(b.State); b.Text != "" || sig != "" {
				out = append(out, part{Text: new(b.Text), Thought: true, ThoughtSignature: sig})
			}
		case litellm.ToolUseBlock:
			args := json.RawMessage("{}")
			if len(b.Arguments) > 0 {
				var object map[string]json.RawMessage
				if json.Unmarshal(b.Arguments, &object) != nil || object == nil {
					return nil, fmt.Errorf("tool use %q arguments must be a JSON object", b.ID)
				}
				args = json.RawMessage(b.Arguments)
			}
			names[b.ID] = b.Name
			out = append(out, part{FunctionCall: &functionCall{ID: b.ID, Name: b.Name, Args: args}, ThoughtSignature: signature(b.State)})
		case litellm.ToolResultBlock:
			name, ok := names[b.ToolUseID]
			if !ok {
				return nil, fmt.Errorf("tool result %q has no preceding tool use", b.ToolUseID)
			}
			response, err := toolResponse(b)
			if err != nil {
				return nil, err
			}
			out = append(out, part{FunctionResponse: &functionResponse{ID: b.ToolUseID, Name: name, Response: response}})
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
	}
	return out, nil
}

// toolResponse wraps the result text as the object Gemini requires: a JSON
// object is sent as is, other text under "result", or "error" on failure.
func toolResponse(result litellm.ToolResultBlock) (json.RawMessage, error) {
	var texts []string
	for _, block := range result.Content {
		text, ok := block.(litellm.TextBlock)
		if !ok {
			return nil, fmt.Errorf("tool results only support text content, got %T", block)
		}
		texts = append(texts, text.Text)
	}
	text := strings.Join(texts, "\n")
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(text), &object) == nil && object != nil {
		return json.RawMessage(text), nil
	}
	key := "result"
	if result.IsError {
		key = "error"
	}
	return json.Marshal(map[string]string{key: text})
}

func convertImage(block litellm.ImageBlock) (part, error) {
	switch {
	case len(block.Data) > 0:
		if block.MIME == "" {
			return part{}, errors.New("inline image requires MIME")
		}
		return part{InlineData: &inlineData{MimeType: block.MIME, Data: base64.StdEncoding.EncodeToString(block.Data)}}, nil
	case block.URL != "":
		if mime, data, ok := wire.ParseDataURL(block.URL); ok {
			return part{InlineData: &inlineData{MimeType: mime, Data: data}}, nil
		}
		return part{FileData: &fileData{MimeType: block.MIME, FileURI: block.URL}}, nil
	case block.FileURI != "":
		return part{FileData: &fileData{MimeType: block.MIME, FileURI: block.FileURI}}, nil
	default:
		return part{}, errors.New("image requires URL, data or file URI")
	}
}

func convertGenerationConfig(req *litellm.Request) (*generationConfig, error) {
	out := &generationConfig{
		Temperature:     req.Temperature,
		MaxOutputTokens: req.MaxTokens,
		TopP:            req.TopP,
		StopSequences:   req.Stop,
		ThinkingConfig:  convertThinking(req.Thinking),
	}
	if format := req.ResponseFormat; format != nil {
		switch format.Type {
		case "", litellm.ResponseFormatText:
		case litellm.ResponseFormatJSONObject:
			out.ResponseMimeType = "application/json"
		case litellm.ResponseFormatJSONSchema:
			out.ResponseMimeType = "application/json"
			if len(format.JSONSchema.Schema) > 0 {
				out.ResponseSchema = json.RawMessage(format.JSONSchema.Schema)
			}
		default:
			return nil, fmt.Errorf("unsupported response format %q", format.Type)
		}
	}
	if out.Temperature == nil && out.MaxOutputTokens == nil && out.TopP == nil && len(out.StopSequences) == 0 &&
		out.ThinkingConfig == nil && out.ResponseMimeType == "" {
		return nil, nil
	}
	return out, nil
}

// convertThinking maps Effort to thinkingLevel, BudgetTokens to
// thinkingBudget and ThinkingDisabled to a zero budget.
func convertThinking(thinking *litellm.Thinking) *thinkingConfig {
	if thinking == nil {
		return nil
	}
	if thinking.Mode == litellm.ThinkingDisabled {
		return &thinkingConfig{ThinkingBudget: new(0)}
	}
	out := &thinkingConfig{ThinkingLevel: thinking.Effort, ThinkingBudget: thinking.BudgetTokens}
	if thinking.IncludeOutput {
		out.IncludeThoughts = true
	}
	return out
}

// convertTools reports whether strict validation was requested. Gemini
// validates all calls or none, so tools cannot mix strict settings.
func convertTools(tools []litellm.Tool) ([]functionDeclaration, bool, error) {
	out := make([]functionDeclaration, 0, len(tools))
	var enabled, disabled bool
	for _, t := range tools {
		enabled = enabled || t.Strict == litellm.StrictEnabled
		disabled = disabled || t.Strict == litellm.StrictDisabled
		declaration := functionDeclaration{Name: t.Name, Description: t.Description}
		if len(t.Parameters) > 0 {
			declaration.ParametersJSONSchema = json.RawMessage(t.Parameters)
		}
		out = append(out, declaration)
	}
	if enabled && disabled {
		return nil, false, errors.New("tools cannot mix strict and non-strict schemas")
	}
	return out, enabled, nil
}

func convertToolChoice(choice *litellm.ToolChoice, strict bool) *toolConfig {
	var mode string
	var allowed []string
	switch {
	case choice == nil:
	case choice.Name != "":
		mode, allowed = "ANY", []string{choice.Name}
	case choice.Mode == litellm.ToolChoiceRequired:
		mode = "ANY"
	default:
		mode = strings.ToUpper(string(choice.Mode))
	}
	if strict && (mode == "" || mode == "AUTO") {
		mode = "VALIDATED"
	}
	if mode == "" {
		return nil
	}
	return &toolConfig{FunctionCallingConfig: &functionCallingConfig{Mode: mode, AllowedFunctionNames: allowed}}
}
