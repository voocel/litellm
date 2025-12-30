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

type bedrockModelList struct {
	ModelSummaries []bedrockModelSummary `json:"modelSummaries"`
}

type bedrockModelSummary struct {
	ModelId          string `json:"modelId"`
	ModelName        string `json:"modelName,omitempty"`
	ProviderName     string `json:"providerName,omitempty"`
	InputTokenLimit  int    `json:"inputTokenLimit,omitempty"`
	OutputTokenLimit int    `json:"outputTokenLimit,omitempty"`
}

func (p *BedrockProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
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
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("bedrock", resp.StatusCode, string(respBody))
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
	if err := p.BaseProvider.ValidateExtra(req.Extra, nil); err != nil {
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
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("bedrock", resp.StatusCode, string(respBody))
	}

	return &bedrockStreamReader{
		reader:   bufio.NewReader(resp.Body),
		response: resp,
		model:    req.Model,
	}, nil
}

// ListModels returns available foundation models for Bedrock.
func (p *BedrockProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}

	baseURL := ""
	if p.Config().Extra != nil {
		if controlPlane, ok := p.Config().Extra["control_plane_base_url"].(string); ok && controlPlane != "" {
			baseURL = strings.TrimSuffix(controlPlane, "/")
		}
	}
	if baseURL == "" {
		baseURL = strings.TrimSuffix(p.Config().BaseURL, "/")
		if baseURL == "" {
			baseURL = fmt.Sprintf("https://bedrock.%s.amazonaws.com", p.region)
		} else if strings.Contains(baseURL, "bedrock-runtime.") {
			baseURL = strings.Replace(baseURL, "bedrock-runtime.", "bedrock.", 1)
		}
	}
	url := fmt.Sprintf("%s/foundation-models", baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("bedrock: create models request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	if err := p.signRequest(httpReq, nil); err != nil {
		return nil, fmt.Errorf("bedrock: sign request: %w", err)
	}

	resp, err := p.HTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, NewHTTPError("bedrock", resp.StatusCode, string(body))
	}

	var payload bedrockModelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("bedrock: decode models response: %w", err)
	}

	models := make([]ModelInfo, 0, len(payload.ModelSummaries))
	for _, item := range payload.ModelSummaries {
		name := item.ModelName
		if name == "" {
			name = item.ModelId
		}
		models = append(models, ModelInfo{
			ID:               item.ModelId,
			Name:             name,
			Provider:         item.ProviderName,
			InputTokenLimit:  item.InputTokenLimit,
			OutputTokenLimit: item.OutputTokenLimit,
		})
	}

	return models, nil
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
