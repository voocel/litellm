package openaicompat

import "encoding/json"

type chatResponse struct {
	Model   string          `json:"model"`
	Choices []choice        `json:"choices"`
	Usage   usage           `json:"usage"`
	Error   json.RawMessage `json:"error"`
}

type choice struct {
	Message      message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Logprobs     json.RawMessage `json:"logprobs"`
}

type message struct {
	Content     json.RawMessage   `json:"content"`
	Refusal     string            `json:"refusal"`
	ToolCalls   []toolCall        `json:"tool_calls"`
	Annotations []json.RawMessage `json:"annotations"`
	// Fields keeps every member so vendor reasoning fields can be read by name.
	Fields map[string]json.RawMessage `json:"-"`
}

func (m *message) UnmarshalJSON(data []byte) error {
	type plain message
	if err := json.Unmarshal(data, (*plain)(m)); err != nil {
		return err
	}
	return json.Unmarshal(data, &m.Fields)
}

type contentPart struct {
	Type        string            `json:"type"`
	Text        string            `json:"text"`
	Refusal     string            `json:"refusal"`
	Annotations []json.RawMessage `json:"annotations"`
	Logprobs    json.RawMessage   `json:"logprobs"`
}

type toolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type usage struct {
	PromptTokens         *int `json:"prompt_tokens"`
	CompletionTokens     *int `json:"completion_tokens"`
	TotalTokens          *int `json:"total_tokens"`
	PromptCacheHitTokens *int `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails  *struct {
		CachedTokens     *int `json:"cached_tokens"`
		CacheWriteTokens *int `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens *int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type streamChunk struct {
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
	Usage   *usage         `json:"usage"`
	// Error is set on mid-stream failures instead of choices.
	Error json.RawMessage `json:"error"`
}

type streamChoice struct {
	Delta        delta         `json:"delta"`
	FinishReason string        `json:"finish_reason"`
	Logprobs     *chatLogprobs `json:"logprobs"`
}

// Tokens stay raw so byte arrays, alternative tokens and vendor extensions
// survive aggregation without reinterpreting their values.
type chatLogprobs struct {
	Content []json.RawMessage `json:"content"`
	Refusal []json.RawMessage `json:"refusal"`
}

type delta struct {
	Content     string                     `json:"content"`
	Refusal     string                     `json:"refusal"`
	ToolCalls   []toolCallDelta            `json:"tool_calls"`
	Annotations []json.RawMessage          `json:"annotations"`
	Fields      map[string]json.RawMessage `json:"-"`
}

func (d *delta) UnmarshalJSON(data []byte) error {
	type plain delta
	if err := json.Unmarshal(data, (*plain)(d)); err != nil {
		return err
	}
	return json.Unmarshal(data, &d.Fields)
}

type toolCallDelta struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type modelList struct {
	Data []struct {
		ID            string `json:"id"`
		Name          string `json:"name"`
		Description   string `json:"description"`
		Created       int64  `json:"created"`
		ContextLength int    `json:"context_length"`
	} `json:"data"`
}
