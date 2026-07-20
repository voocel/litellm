package otel

import (
	"encoding/base64"
	"encoding/json"

	"github.com/voocel/litellm"
)

type genAIMessage struct {
	Role  string `json:"role"`
	Parts []any  `json:"parts"`
}

type genAIOutputMessage struct {
	Role         string `json:"role"`
	Parts        []any  `json:"parts"`
	FinishReason string `json:"finish_reason"`
}

type genAITextPart struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type genAIReasoningPart struct {
	Type    string `json:"type"`
	Content string `json:"content"`
}

type genAIBlobPart struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Modality string `json:"modality"`
	Content  string `json:"content"`
}

type genAIURIPart struct {
	Type     string `json:"type"`
	MIMEType string `json:"mime_type,omitempty"`
	Modality string `json:"modality"`
	URI      string `json:"uri"`
}

type genAIToolCallPart struct {
	Type      string `json:"type"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

type genAIToolCallResponsePart struct {
	Type     string `json:"type"`
	ID       string `json:"id,omitempty"`
	Response any    `json:"response"`
	IsError  bool   `json:"is_error,omitempty"`
}

type genAIToolReferencePart struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

func marshalInputMessages(messages []litellm.Message) (string, error) {
	encoded := make([]genAIMessage, 0, len(messages))
	for _, message := range messages {
		encoded = append(encoded, genAIMessage{
			Role:  string(message.Role),
			Parts: genAIParts(message.Blocks),
		})
	}
	data, err := json.Marshal(encoded)
	return string(data), err
}

func marshalOutputMessages(blocks []litellm.Block, finishReason litellm.FinishReason) (string, error) {
	data, err := json.Marshal([]genAIOutputMessage{{
		Role:         string(litellm.RoleAssistant),
		Parts:        genAIParts(blocks),
		FinishReason: semanticFinishReason(finishReason),
	}})
	return string(data), err
}

func genAIParts(blocks []litellm.Block) []any {
	parts := make([]any, 0, len(blocks))
	for _, block := range blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			parts = append(parts, genAITextPart{Type: "text", Content: b.Text})
		case litellm.ImageBlock:
			parts = append(parts, genAIImagePart(b))
		case litellm.ReasoningBlock:
			if b.Text != "" {
				parts = append(parts, genAIReasoningPart{Type: "reasoning", Content: b.Text})
			}
		case litellm.ToolUseBlock:
			parts = append(parts, genAIToolCallPart{
				Type:      "tool_call",
				ID:        b.ID,
				Name:      b.Name,
				Arguments: rawJSONValue(b.Arguments),
			})
		case litellm.ToolResultBlock:
			parts = append(parts, genAIToolCallResponsePart{
				Type:     "tool_call_response",
				ID:       b.ToolUseID,
				Response: toolResponse(b.Content),
				IsError:  b.IsError,
			})
		case litellm.ToolReferenceBlock:
			parts = append(parts, genAIToolReferencePart{Type: "tool_reference", Name: b.ToolName})
		}
	}
	return parts
}

func genAIImagePart(image litellm.ImageBlock) any {
	if len(image.Data) > 0 {
		return genAIBlobPart{
			Type:     "blob",
			MIMEType: image.MIME,
			Modality: "image",
			Content:  base64.StdEncoding.EncodeToString(image.Data),
		}
	}
	uri := image.FileURI
	if uri == "" {
		uri = image.URL
	}
	return genAIURIPart{
		Type:     "uri",
		MIMEType: image.MIME,
		Modality: "image",
		URI:      uri,
	}
}

func toolResponse(blocks []litellm.Block) any {
	parts := genAIParts(blocks)
	if len(parts) == 1 {
		if text, ok := parts[0].(genAITextPart); ok {
			return text.Content
		}
	}
	return parts
}

func rawJSONValue(data json.RawMessage) any {
	if len(data) == 0 {
		return nil
	}
	var value any
	if json.Unmarshal(data, &value) != nil {
		return string(data)
	}
	return value
}
