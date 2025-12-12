package providers

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

func init() {
	RegisterBuiltin("bedrock", func(cfg ProviderConfig) Provider {
		return NewBedrock(cfg)
	}, "") // Base URL is region-specific
}

type BedrockProvider struct {
	*BaseProvider
	region          string
	accessKeyID     string
	secretAccessKey string
	sessionToken    string
}

func NewBedrock(config ProviderConfig) *BedrockProvider {
	region := "us-east-1"
	var accessKeyID, secretAccessKey, sessionToken string

	if config.Extra != nil {
		if r, ok := config.Extra["region"].(string); ok && r != "" {
			region = r
		}
		if ak, ok := config.Extra["access_key_id"].(string); ok {
			accessKeyID = ak
		}
		if sk, ok := config.Extra["secret_access_key"].(string); ok {
			secretAccessKey = sk
		}
		if st, ok := config.Extra["session_token"].(string); ok {
			sessionToken = st
		}
	}

	if accessKeyID == "" && config.APIKey != "" {
		parts := strings.SplitN(config.APIKey, ":", 2)
		if len(parts) == 2 {
			accessKeyID = parts[0]
			secretAccessKey = parts[1]
		}
	}

	if config.BaseURL == "" {
		config.BaseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region)
	}

	baseProvider := NewBaseProvider("bedrock", config)

	return &BedrockProvider{
		BaseProvider:    baseProvider,
		region:          region,
		accessKeyID:     accessKeyID,
		secretAccessKey: secretAccessKey,
		sessionToken:    sessionToken,
	}
}

func (p *BedrockProvider) Validate() error {
	if p.accessKeyID == "" {
		return fmt.Errorf("bedrock: AWS access key ID is required")
	}
	if p.secretAccessKey == "" {
		return fmt.Errorf("bedrock: AWS secret access key is required")
	}
	return nil
}

func (p *BedrockProvider) SupportsModel(model string) bool {
	supportedPrefixes := []string{
		"anthropic.", "amazon.", "meta.", "mistral.",
		"cohere.", "ai21.", "stability.", "deepseek.", "qwen.", "google.",
	}
	for _, prefix := range supportedPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	for _, m := range p.Models() {
		if m.ID == model {
			return true
		}
	}
	return false
}

func (p *BedrockProvider) Models() []ModelInfo {
	return []ModelInfo{
		// Claude 4.5
		{
			ID: "anthropic.claude-opus-4-5-20251101-v1:0", Provider: "bedrock", Name: "Claude Opus 4.5",
			ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		{
			ID: "anthropic.claude-sonnet-4-5-20250929-v1:0", Provider: "bedrock", Name: "Claude Sonnet 4.5",
			ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		{
			ID: "anthropic.claude-haiku-4-5-20251001-v1:0", Provider: "bedrock", Name: "Claude Haiku 4.5",
			ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "vision", "function_call"},
		},
		// Claude 4
		{
			ID: "anthropic.claude-opus-4-1-20250805-v1:0", Provider: "bedrock", Name: "Claude Opus 4.1",
			ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		{
			ID: "anthropic.claude-opus-4-20250514-v1:0", Provider: "bedrock", Name: "Claude Opus 4",
			ContextWindow: 200000, MaxOutputTokens: 32000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		{
			ID: "anthropic.claude-sonnet-4-20250514-v1:0", Provider: "bedrock", Name: "Claude Sonnet 4",
			ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		// Claude 3.7/3.5/3
		{
			ID: "anthropic.claude-3-7-sonnet-20250219-v1:0", Provider: "bedrock", Name: "Claude 3.7 Sonnet",
			ContextWindow: 200000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "vision", "function_call", "extended_thinking"},
		},
		{
			ID: "anthropic.claude-3-5-haiku-20241022-v1:0", Provider: "bedrock", Name: "Claude 3.5 Haiku",
			ContextWindow: 200000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "anthropic.claude-3-haiku-20240307-v1:0", Provider: "bedrock", Name: "Claude 3 Haiku",
			ContextWindow: 200000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "vision", "function_call"},
		},
		// Amazon Nova
		{
			ID: "amazon.nova-premier-v1:0", Provider: "bedrock", Name: "Amazon Nova Premier",
			ContextWindow: 1000000, MaxOutputTokens: 5000,
			Capabilities: []string{"chat", "vision", "video", "function_call", "reasoning"},
		},
		{
			ID: "amazon.nova-2-lite-v1:0", Provider: "bedrock", Name: "Amazon Nova 2 Lite",
			ContextWindow: 300000, MaxOutputTokens: 5000,
			Capabilities: []string{"chat", "vision", "video", "function_call"},
		},
		{
			ID: "amazon.nova-pro-v1:0", Provider: "bedrock", Name: "Amazon Nova Pro",
			ContextWindow: 300000, MaxOutputTokens: 5000,
			Capabilities: []string{"chat", "vision", "video", "function_call", "reasoning"},
		},
		{
			ID: "amazon.nova-lite-v1:0", Provider: "bedrock", Name: "Amazon Nova Lite",
			ContextWindow: 300000, MaxOutputTokens: 5000,
			Capabilities: []string{"chat", "vision", "video", "function_call"},
		},
		{
			ID: "amazon.nova-micro-v1:0", Provider: "bedrock", Name: "Amazon Nova Micro",
			ContextWindow: 128000, MaxOutputTokens: 5000,
			Capabilities: []string{"chat", "function_call"},
		},
		// Meta Llama
		{
			ID: "meta.llama4-maverick-17b-instruct-v1:0", Provider: "bedrock", Name: "Llama 4 Maverick 17B",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "meta.llama4-scout-17b-instruct-v1:0", Provider: "bedrock", Name: "Llama 4 Scout 17B",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "meta.llama3-3-70b-instruct-v1:0", Provider: "bedrock", Name: "Llama 3.3 70B Instruct",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "meta.llama3-2-90b-instruct-v1:0", Provider: "bedrock", Name: "Llama 3.2 90B Instruct",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "vision", "function_call"},
		},
		{
			ID: "meta.llama3-1-405b-instruct-v1:0", Provider: "bedrock", Name: "Llama 3.1 405B Instruct",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "meta.llama3-1-70b-instruct-v1:0", Provider: "bedrock", Name: "Llama 3.1 70B Instruct",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		// Mistral
		{
			ID: "mistral.mistral-large-2411-v1:0", Provider: "bedrock", Name: "Mistral Large 3",
			ContextWindow: 128000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "mistral.pixtral-large-2502-v1:0", Provider: "bedrock", Name: "Pixtral Large",
			ContextWindow: 128000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "vision", "function_call"},
		},
		{
			ID: "mistral.magistral-small-2509-v1:0", Provider: "bedrock", Name: "Magistral Small",
			ContextWindow: 128000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "function_call"},
		},
		// DeepSeek
		{
			ID: "deepseek.r1-v1:0", Provider: "bedrock", Name: "DeepSeek R1",
			ContextWindow: 128000, MaxOutputTokens: 64000,
			Capabilities: []string{"chat", "reasoning"},
		},
		{
			ID: "deepseek.v3-v1:0", Provider: "bedrock", Name: "DeepSeek V3.1",
			ContextWindow: 128000, MaxOutputTokens: 8000,
			Capabilities: []string{"chat", "function_call"},
		},
		// Qwen
		{
			ID: "qwen.qwen3-235b-a22b-2507-v1:0", Provider: "bedrock", Name: "Qwen3 235B",
			ContextWindow: 128000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "function_call", "reasoning"},
		},
		{
			ID: "qwen.qwen3-32b-instruct-v1:0", Provider: "bedrock", Name: "Qwen3 32B Instruct",
			ContextWindow: 128000, MaxOutputTokens: 8192,
			Capabilities: []string{"chat", "function_call"},
		},
		// Cohere
		{
			ID: "cohere.command-r-plus-v1:0", Provider: "bedrock", Name: "Command R+",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
		{
			ID: "cohere.command-r-v1:0", Provider: "bedrock", Name: "Command R",
			ContextWindow: 128000, MaxOutputTokens: 4096,
			Capabilities: []string{"chat", "function_call"},
		},
	}
}

type bedrockRequest struct {
	Messages                     []bedrockMessage        `json:"messages"`
	System                       []bedrockSystemContent  `json:"system,omitempty"`
	InferenceConfig              *bedrockInferenceConfig `json:"inferenceConfig,omitempty"`
	ToolConfig                   *bedrockToolConfig      `json:"toolConfig,omitempty"`
	AdditionalModelRequestFields map[string]any          `json:"additionalModelRequestFields,omitempty"`
}

type bedrockMessage struct {
	Role    string           `json:"role"`
	Content []bedrockContent `json:"content"`
}

type bedrockContent struct {
	Text       string               `json:"text,omitempty"`
	Image      *bedrockImageContent `json:"image,omitempty"`
	ToolUse    *bedrockToolUse      `json:"toolUse,omitempty"`
	ToolResult *bedrockToolResult   `json:"toolResult,omitempty"`
}

type bedrockImageContent struct {
	Format string             `json:"format"`
	Source bedrockImageSource `json:"source"`
}

type bedrockImageSource struct {
	Bytes string `json:"bytes,omitempty"`
}

type bedrockToolUse struct {
	ToolUseID string `json:"toolUseId"`
	Name      string `json:"name"`
	Input     any    `json:"input"`
}

type bedrockToolResult struct {
	ToolUseID string           `json:"toolUseId"`
	Content   []bedrockContent `json:"content"`
	Status    string           `json:"status,omitempty"`
}

type bedrockSystemContent struct {
	Text string `json:"text"`
}

type bedrockInferenceConfig struct {
	MaxTokens     int      `json:"maxTokens,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopP          float64  `json:"topP,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type bedrockToolConfig struct {
	Tools      []bedrockTool `json:"tools"`
	ToolChoice any           `json:"toolChoice,omitempty"`
}

type bedrockTool struct {
	ToolSpec *bedrockToolSpec `json:"toolSpec,omitempty"`
}

type bedrockToolSpec struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema"`
}

type bedrockResponse struct {
	Output struct {
		Message bedrockMessage `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      struct {
		InputTokens  int `json:"inputTokens"`
		OutputTokens int `json:"outputTokens"`
		TotalTokens  int `json:"totalTokens"`
	} `json:"usage"`
	Metrics struct {
		LatencyMs int64 `json:"latencyMs"`
	} `json:"metrics"`
}

func (p *BedrockProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	bedrockReq := p.buildRequest(req)

	body, err := json.Marshal(bedrockReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/converse", p.Config().BaseURL, req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: create request: %w", err)
	}

	if err := p.signRequest(httpReq, body); err != nil {
		return nil, fmt.Errorf("bedrock: sign request: %w", err)
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: API error %d: %s", resp.StatusCode, string(respBody))
	}

	var bedrockResp bedrockResponse
	if err := json.NewDecoder(resp.Body).Decode(&bedrockResp); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}

	return p.parseResponse(&bedrockResp, req.Model), nil
}

func (p *BedrockProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateRequest(req); err != nil {
		return nil, err
	}

	bedrockReq := p.buildRequest(req)

	body, err := json.Marshal(bedrockReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/model/%s/converse-stream", p.Config().BaseURL, req.Model)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock: create request: %w", err)
	}

	if err := p.signRequest(httpReq, body); err != nil {
		return nil, fmt.Errorf("bedrock: sign request: %w", err)
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("bedrock: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock: API error %d: %s", resp.StatusCode, string(respBody))
	}

	return &bedrockStreamReader{
		reader:   bufio.NewReader(resp.Body),
		response: resp,
		model:    req.Model,
	}, nil
}

func (p *BedrockProvider) buildRequest(req *Request) *bedrockRequest {
	bedrockReq := &bedrockRequest{
		Messages: make([]bedrockMessage, 0, len(req.Messages)),
	}

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			bedrockReq.System = append(bedrockReq.System, bedrockSystemContent{Text: msg.Content})
			continue
		}

		bedrockMsg := bedrockMessage{
			Role:    msg.Role,
			Content: []bedrockContent{},
		}

		if msg.Role == "tool" && msg.ToolCallID != "" {
			bedrockMsg.Role = "user"
			bedrockMsg.Content = append(bedrockMsg.Content, bedrockContent{
				ToolResult: &bedrockToolResult{
					ToolUseID: msg.ToolCallID,
					Content:   []bedrockContent{{Text: msg.Content}},
				},
			})
		} else if msg.Content != "" {
			bedrockMsg.Content = append(bedrockMsg.Content, bedrockContent{Text: msg.Content})
		}

		for _, tc := range msg.ToolCalls {
			var input any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				input = map[string]any{}
			}
			bedrockMsg.Content = append(bedrockMsg.Content, bedrockContent{
				ToolUse: &bedrockToolUse{
					ToolUseID: tc.ID,
					Name:      tc.Function.Name,
					Input:     input,
				},
			})
		}

		bedrockReq.Messages = append(bedrockReq.Messages, bedrockMsg)
	}

	if req.MaxTokens != nil || req.Temperature != nil || len(req.Stop) > 0 {
		bedrockReq.InferenceConfig = &bedrockInferenceConfig{}
		if req.MaxTokens != nil {
			bedrockReq.InferenceConfig.MaxTokens = *req.MaxTokens
		}
		if req.Temperature != nil {
			bedrockReq.InferenceConfig.Temperature = *req.Temperature
		}
		if req.TopP != nil {
			bedrockReq.InferenceConfig.TopP = *req.TopP
		}
		if len(req.Stop) > 0 {
			bedrockReq.InferenceConfig.StopSequences = req.Stop
		}
	}

	if len(req.Tools) > 0 {
		bedrockReq.ToolConfig = &bedrockToolConfig{
			Tools: make([]bedrockTool, len(req.Tools)),
		}
		for i, tool := range req.Tools {
			bedrockReq.ToolConfig.Tools[i] = bedrockTool{
				ToolSpec: &bedrockToolSpec{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					InputSchema: map[string]any{"json": tool.Function.Parameters},
				},
			}
		}
		if req.ToolChoice != nil {
			bedrockReq.ToolConfig.ToolChoice = req.ToolChoice
		}
	}

	return bedrockReq
}

func (p *BedrockProvider) parseResponse(resp *bedrockResponse, model string) *Response {
	response := &Response{
		Model:        model,
		Provider:     "bedrock",
		FinishReason: mapStopReason(resp.StopReason),
		Usage: Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	for _, content := range resp.Output.Message.Content {
		if content.Text != "" {
			response.Content += content.Text
		}
		if content.ToolUse != nil {
			inputJSON, _ := json.Marshal(content.ToolUse.Input)
			response.ToolCalls = append(response.ToolCalls, ToolCall{
				ID:   content.ToolUse.ToolUseID,
				Type: "function",
				Function: FunctionCall{
					Name:      content.ToolUse.Name,
					Arguments: string(inputJSON),
				},
			})
		}
	}

	return response
}

func mapStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "stop_sequence":
		return "stop"
	default:
		return reason
	}
}

// AWS Signature V4 implementation
func (p *BedrockProvider) signRequest(req *http.Request, payload []byte) error {
	now := time.Now().UTC()
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Amz-Date", amzDate)
	if p.sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", p.sessionToken)
	}

	payloadHash := sha256Hex(payload)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)

	canonicalURI := req.URL.Path
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQueryString := req.URL.RawQuery

	signedHeaders := []string{"content-type", "host", "x-amz-content-sha256", "x-amz-date"}
	if p.sessionToken != "" {
		signedHeaders = append(signedHeaders, "x-amz-security-token")
	}
	sort.Strings(signedHeaders)

	canonicalHeaders := ""
	for _, h := range signedHeaders {
		var val string
		switch h {
		case "host":
			val = req.URL.Host
		default:
			val = req.Header.Get(h)
		}
		canonicalHeaders += h + ":" + strings.TrimSpace(val) + "\n"
	}

	signedHeadersStr := strings.Join(signedHeaders, ";")

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQueryString,
		canonicalHeaders,
		signedHeadersStr,
		payloadHash,
	}, "\n")

	algorithm := "AWS4-HMAC-SHA256"
	credentialScope := dateStamp + "/" + p.region + "/bedrock/aws4_request"
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	signingKey := getSignatureKey(p.secretAccessKey, dateStamp, p.region, "bedrock")
	signature := hmacSHA256Hex(signingKey, []byte(stringToSign))

	authHeader := fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		algorithm, p.accessKeyID, credentialScope, signedHeadersStr, signature)
	req.Header.Set("Authorization", authHeader)

	return nil
}

func sha256Hex(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func hmacSHA256Hex(key, data []byte) string {
	return hex.EncodeToString(hmacSHA256(key, data))
}

func getSignatureKey(secretKey, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secretKey), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	return kSigning
}

type bedrockStreamReader struct {
	reader   *bufio.Reader
	response *http.Response
	model    string
	done     bool
}

func (s *bedrockStreamReader) Next() (*StreamChunk, error) {
	if s.done {
		return &StreamChunk{Done: true, Provider: "bedrock", Model: s.model}, nil
	}

	// Read event stream format
	// AWS uses binary event stream encoding, simplified parsing here
	for {
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				s.done = true
				return &StreamChunk{Done: true, Provider: "bedrock", Model: s.model}, nil
			}
			return nil, fmt.Errorf("bedrock: read stream: %w", err)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		var event map[string]json.RawMessage
		if err := json.Unmarshal(line, &event); err != nil {
			continue
		}

		if data, ok := event["contentBlockDelta"]; ok {
			var delta struct {
				Delta struct {
					Text string `json:"text"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(data, &delta); err == nil && delta.Delta.Text != "" {
				return &StreamChunk{
					Type:     "content",
					Content:  delta.Delta.Text,
					Provider: "bedrock",
					Model:    s.model,
				}, nil
			}
		}

		if data, ok := event["messageStop"]; ok {
			var stop struct {
				StopReason string `json:"stopReason"`
			}
			_ = json.Unmarshal(data, &stop)
			s.done = true
			return &StreamChunk{
				Done:         true,
				FinishReason: mapStopReason(stop.StopReason),
				Provider:     "bedrock",
				Model:        s.model,
			}, nil
		}

		if data, ok := event["metadata"]; ok {
			var meta struct {
				Usage struct {
					InputTokens  int `json:"inputTokens"`
					OutputTokens int `json:"outputTokens"`
					TotalTokens  int `json:"totalTokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(data, &meta); err == nil {
				return &StreamChunk{
					Type:     "metadata",
					Provider: "bedrock",
					Model:    s.model,
					Usage: &Usage{
						PromptTokens:     meta.Usage.InputTokens,
						CompletionTokens: meta.Usage.OutputTokens,
						TotalTokens:      meta.Usage.TotalTokens,
					},
				}, nil
			}
		}
	}
}

func (s *bedrockStreamReader) Close() error {
	return s.response.Body.Close()
}
