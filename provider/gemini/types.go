package gemini

import "encoding/json"

type request struct {
	Contents          []content         `json:"contents"`
	GenerationConfig  *generationConfig `json:"generationConfig,omitempty"`
	Tools             []tool            `json:"tools,omitempty"`
	ToolConfig        *toolConfig       `json:"toolConfig,omitempty"`
	SystemInstruction *content          `json:"systemInstruction,omitempty"`
}

type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	// Text is a pointer because text is the part's data: a signature-only
	// part must still send "text": "".
	Text             *string           `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type functionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type functionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type generationConfig struct {
	Temperature     *float64              `json:"temperature,omitempty"`
	MaxOutputTokens *int                  `json:"maxOutputTokens,omitempty"`
	TopP            *float64              `json:"topP,omitempty"`
	StopSequences   []string              `json:"stopSequences,omitempty"`
	ResponseFormat  *responseFormatConfig `json:"responseFormat,omitempty"`
	ThinkingConfig  *thinkingConfig       `json:"thinkingConfig,omitempty"`
}

type responseFormatConfig struct {
	Text textResponseFormat `json:"text"`
}

type textResponseFormat struct {
	// The REST schema defines an enum here, unlike the legacy MIME string.
	MimeType string          `json:"mimeType"`
	Schema   json.RawMessage `json:"schema,omitempty"`
}

type thinkingConfig struct {
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
	IncludeThoughts bool   `json:"includeThoughts,omitempty"`
}

type tool struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name                 string          `json:"name"`
	Description          string          `json:"description,omitempty"`
	ParametersJSONSchema json.RawMessage `json:"parametersJsonSchema,omitempty"`
}

type toolConfig struct {
	FunctionCallingConfig *functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

type functionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type response struct {
	Candidates     []candidate     `json:"candidates"`
	UsageMetadata  *usageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *promptFeedback `json:"promptFeedback,omitempty"`
	// Error is set on a stream chunk that reports a failure mid-stream.
	Error json.RawMessage `json:"error,omitempty"`
}

type candidate struct {
	Content       content `json:"content"`
	FinishReason  string  `json:"finishReason,omitempty"`
	FinishMessage string  `json:"finishMessage,omitempty"`
}

// usageMetadata counts are plain ints: the API omits zero-valued fields.
type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

type promptFeedback struct {
	BlockReason string `json:"blockReason"`
}

type modelList struct {
	Models        []modelInfo `json:"models"`
	NextPageToken string      `json:"nextPageToken,omitempty"`
}

type modelInfo struct {
	Name             string `json:"name"`
	DisplayName      string `json:"displayName,omitempty"`
	Description      string `json:"description,omitempty"`
	InputTokenLimit  int    `json:"inputTokenLimit,omitempty"`
	OutputTokenLimit int    `json:"outputTokenLimit,omitempty"`
}
