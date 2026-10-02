package bedrock

import "encoding/json"

type request struct {
	Messages                     []message        `json:"messages"`
	System                       []content        `json:"system,omitempty"`
	InferenceConfig              *inferenceConfig `json:"inferenceConfig,omitempty"`
	ToolConfig                   *toolConfig      `json:"toolConfig,omitempty"`
	OutputConfig                 *outputConfig    `json:"outputConfig,omitempty"`
	AdditionalModelRequestFields map[string]any   `json:"additionalModelRequestFields,omitempty"`
}

type message struct {
	Role    string    `json:"role"`
	Content []content `json:"content"`
}

type content struct {
	Text             string            `json:"text,omitempty"`
	Image            *image            `json:"image,omitempty"`
	ReasoningContent *reasoningContent `json:"reasoningContent,omitempty"`
	ToolUse          *toolUse          `json:"toolUse,omitempty"`
	ToolResult       *toolResult       `json:"toolResult,omitempty"`
	CachePoint       *cachePoint       `json:"cachePoint,omitempty"`
}

type cachePoint struct {
	Type string `json:"type"`
}

type reasoningContent struct {
	ReasoningText   *reasoningText `json:"reasoningText,omitempty"`
	RedactedContent []byte         `json:"redactedContent,omitempty"`
}

type reasoningText struct {
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

// reasoningState is the ProviderState of a reasoning block: the signature or
// encrypted content needed to send it back.
type reasoningState struct {
	Signature       string `json:"signature,omitempty"`
	RedactedContent []byte `json:"redactedContent,omitempty"`
}

type image struct {
	Format string      `json:"format"`
	Source imageSource `json:"source"`
}

type imageSource struct {
	Bytes []byte `json:"bytes,omitempty"`
}

type toolUse struct {
	ToolUseID string          `json:"toolUseId"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
}

type toolResult struct {
	ToolUseID string    `json:"toolUseId"`
	Content   []content `json:"content"`
	Status    string    `json:"status,omitempty"`
}

type inferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type toolConfig struct {
	Tools      []tool `json:"tools"`
	ToolChoice any    `json:"toolChoice,omitempty"`
}

type tool struct {
	ToolSpec *toolSpec `json:"toolSpec,omitempty"`
}

type toolSpec struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Strict      *bool       `json:"strict,omitempty"`
	InputSchema inputSchema `json:"inputSchema"`
}

type inputSchema struct {
	JSON json.RawMessage `json:"json"`
}

type outputConfig struct {
	TextFormat *textFormat `json:"textFormat,omitempty"`
}

type textFormat struct {
	Type      string              `json:"type"`
	Structure textFormatStructure `json:"structure"`
}

type textFormatStructure struct {
	JSONSchema jsonSchema `json:"jsonSchema"`
}

type jsonSchema struct {
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Schema      string `json:"schema"`
}

type usage struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
}

type response struct {
	Output struct {
		Message message `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      usage  `json:"usage"`
}
