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
// naming a generated object is merged into it, e.g. {"topK": 40} under
// "generationConfig". Gemini's built-in tools are not offered: their output
// parts have no block to land in.
const (
	ProviderOptionSafetySettings   = "safetySettings"
	ProviderOptionGenerationConfig = "generationConfig"
	ProviderOptionToolConfig       = "toolConfig"
	ProviderOptionCachedContent    = "cachedContent"
)

var providerOptions = []string{
	ProviderOptionSafetySettings, ProviderOptionGenerationConfig,
	ProviderOptionToolConfig, ProviderOptionCachedContent,
}

func sortedOptions() []string {
	out := slices.Clone(providerOptions)
	slices.Sort(out)
	return out
}

func buildRequest(req *litellm.Request, provider string) ([]byte, error) {
	opts, err := req.ProviderOptions.Decode()
	if err != nil {
		return nil, err
	}
	if err := wire.CheckOptions(opts, providerOptions); err != nil {
		return nil, err
	}
	if config, ok := opts[ProviderOptionGenerationConfig].(map[string]any); ok {
		if count, exists := config["candidateCount"]; exists {
			number, ok := count.(json.Number)
			n, err := number.Float64()
			if !ok || err != nil || n != 1 {
				return nil, errors.New("generationConfig.candidateCount must be 1; litellm.Response holds a single output")
			}
		}
	}
	out := &request{}
	contents, system, err := convertMessages(req.Messages, provider)
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
	if offered := req.OfferedTools(); len(offered) > 0 {
		declarations, strict, err := convertTools(offered)
		if err != nil {
			return nil, err
		}
		out.Tools = []tool{{FunctionDeclarations: declarations}}
		out.ToolConfig = convertToolChoice(req.ToolChoice, strict)
	}
	return wire.MarshalBody(out, opts)
}

func convertMessages(messages []litellm.Message, provider string) ([]content, []part, error) {
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
			parts, err = convertBlocks(msg.Blocks, names, provider)
			system = append(system, parts...)
			if err != nil {
				return nil, nil, fmt.Errorf("messages[%d]: %w", i, err)
			}
			continue
		case litellm.RoleAssistant:
			role = "model"
		}
		if parts, err = convertBlocks(msg.Blocks, names, provider); err != nil {
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

func convertBlocks(blocks []litellm.Block, names map[string]string, provider string) ([]part, error) {
	out := make([]part, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			if sig := signature(b.State, provider); b.Text != "" || sig != "" {
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
			if sig := signature(b.State, provider); b.Text != "" || sig != "" {
				out = append(out, part{Text: new(b.Text), Thought: true, ThoughtSignature: sig})
			}
		case litellm.ToolUseBlock:
			args := json.RawMessage("{}")
			if b.Arguments != "" {
				var object map[string]json.RawMessage
				if json.Unmarshal([]byte(b.Arguments), &object) != nil || object == nil {
					return nil, fmt.Errorf("tool use %q (%s) arguments are not a JSON object", b.ID, b.Name)
				}
				args = json.RawMessage(b.Arguments)
			}
			names[b.ID] = b.Name
			out = append(out, part{FunctionCall: &functionCall{ID: b.ID, Name: b.Name, Args: args}, ThoughtSignature: signature(b.State, provider)})
		case litellm.ToolResultBlock:
			name, ok := names[b.ToolUseID]
			if !ok {
				return nil, fmt.Errorf("tool result %q has no preceding tool use", b.ToolUseID)
			}
			response, media, err := toolResponse(b)
			if err != nil {
				return nil, err
			}
			out = append(out, part{FunctionResponse: &functionResponse{ID: b.ToolUseID, Name: name, Response: response, Parts: media}})
		default:
			return nil, fmt.Errorf("unsupported block %T", block)
		}
	}
	return out, nil
}

// toolResponse wraps the result text as the object Gemini requires: a JSON
// object is sent as is on success, other text under "result". Failures always
// go under "error", including JSON objects, without losing numeric precision.
// Images go in the response's media parts, a feature of the Gemini 3 series,
// which takes inline data only.
func toolResponse(result litellm.ToolResultBlock) (json.RawMessage, []functionResponsePart, error) {
	var texts []string
	var media []functionResponsePart
	for _, block := range result.Content {
		switch b := block.(type) {
		case litellm.TextBlock:
			texts = append(texts, b.Text)
		case litellm.ToolReferenceBlock:
			texts = append(texts, wire.ToolReferenceText(b))
		case litellm.ImageBlock:
			image, err := convertImage(b)
			if err != nil {
				return nil, nil, err
			}
			if image.InlineData == nil {
				return nil, nil, errors.New("tool result images must be inline data")
			}
			media = append(media, functionResponsePart{InlineData: image.InlineData})
		default:
			return nil, nil, fmt.Errorf("tool results do not support %T", block)
		}
	}
	text := strings.Join(texts, "\n")
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(text), &object) == nil && object != nil {
		if result.IsError {
			response, err := json.Marshal(map[string]json.RawMessage{"error": json.RawMessage(text)})
			return response, media, err
		}
		return json.RawMessage(text), media, nil
	}
	key := "result"
	if result.IsError {
		key = "error"
	}
	response, err := json.Marshal(map[string]string{key: text})
	return response, media, err
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
			out.ResponseFormat = &responseFormatConfig{Text: textResponseFormat{MimeType: "APPLICATION_JSON"}}
		case litellm.ResponseFormatJSONSchema:
			out.ResponseFormat = &responseFormatConfig{Text: textResponseFormat{
				MimeType: "APPLICATION_JSON",
				Schema:   json.RawMessage(format.JSONSchema.Schema),
			}}
		default:
			return nil, fmt.Errorf("unsupported response format %q", format.Type)
		}
	}
	if out.Temperature == nil && out.MaxOutputTokens == nil && out.TopP == nil && len(out.StopSequences) == 0 &&
		out.ThinkingConfig == nil && out.ResponseFormat == nil {
		return nil, nil
	}
	return out, nil
}

// convertThinking maps Effort to thinkingLevel, BudgetTokens to
// thinkingBudget and Thinking.Disabled to a zero budget.
func convertThinking(thinking *litellm.Thinking) *thinkingConfig {
	if thinking == nil {
		return nil
	}
	if thinking.Disabled {
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
		enabled = enabled || t.Strict != nil && *t.Strict
		disabled = disabled || t.Strict != nil && !*t.Strict
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
