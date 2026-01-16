package litellm

import (
	"fmt"
	"sort"
	"strings"
)

// StreamCallbacks provides optional per-chunk handlers during stream collection.
type StreamCallbacks struct {
	OnChunk     func(*StreamChunk)
	OnContent   func(string)
	OnReasoning func(*ReasoningChunk)
	OnToolCall  func(*ToolCallDelta)
}

// CollectStream consumes a StreamReader and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStream(stream StreamReader) (*Response, error) {
	return CollectStreamWithHandler(stream, nil)
}

// CollectStreamWithCallbacks consumes a StreamReader, calls callbacks for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithCallbacks(stream StreamReader, callbacks StreamCallbacks) (*Response, error) {
	return CollectStreamWithHandler(stream, func(chunk *StreamChunk) {
		if callbacks.OnChunk != nil {
			callbacks.OnChunk(chunk)
		}
		if callbacks.OnContent != nil && chunk.Type == ChunkTypeContent && chunk.Content != "" {
			callbacks.OnContent(chunk.Content)
		}
		if callbacks.OnReasoning != nil && chunk.Reasoning != nil {
			callbacks.OnReasoning(chunk.Reasoning)
		}
		if callbacks.OnToolCall != nil && chunk.ToolCallDelta != nil {
			callbacks.OnToolCall(chunk.ToolCallDelta)
		}
	})
}

// CollectStreamWithHandler consumes a StreamReader, calls onChunk for each chunk, and returns a unified Response.
// Callers are responsible for closing the stream.
func CollectStreamWithHandler(stream StreamReader, onChunk func(*StreamChunk)) (*Response, error) {
	if stream == nil {
		return nil, fmt.Errorf("stream cannot be nil")
	}

	var (
		contentBuilder        strings.Builder
		refusalBuilder        strings.Builder
		contentByOutputIndex  = map[int]*strings.Builder{}
		refusalByOutputIndex  = map[int]*strings.Builder{}
		reasoningSummary      strings.Builder
		reasoningContent      strings.Builder
		toolCallOrder         []string
		toolCallsByIdentifier = map[string]*ToolCall{}
		resp                  Response
	)

	for {
		chunk, err := stream.Next()
		if err != nil {
			return nil, err
		}
		if chunk == nil {
			continue
		}

		if onChunk != nil {
			onChunk(chunk)
		}

		if resp.Provider == "" && chunk.Provider != "" {
			resp.Provider = chunk.Provider
		}
		if chunk.Model != "" {
			resp.Model = chunk.Model
		}
		if resp.FinishReason == "" && chunk.FinishReason != "" {
			resp.FinishReason = chunk.FinishReason
		}

		if chunk.Content != "" {
			switch chunk.Type {
			case ChunkTypeContent:
				if chunk.OutputIndex != nil {
					builder := contentByOutputIndex[*chunk.OutputIndex]
					if builder == nil {
						builder = &strings.Builder{}
						contentByOutputIndex[*chunk.OutputIndex] = builder
					}
					builder.WriteString(chunk.Content)
				} else {
					contentBuilder.WriteString(chunk.Content)
				}
			case "refusal":
				if chunk.OutputIndex != nil {
					builder := refusalByOutputIndex[*chunk.OutputIndex]
					if builder == nil {
						builder = &strings.Builder{}
						refusalByOutputIndex[*chunk.OutputIndex] = builder
					}
					builder.WriteString(chunk.Content)
				} else {
					refusalBuilder.WriteString(chunk.Content)
				}
			}
		}

		if chunk.Reasoning != nil {
			if chunk.Reasoning.Summary != "" {
				if reasoningSummary.Len() > 0 {
					reasoningSummary.WriteString("\n")
				}
				reasoningSummary.WriteString(chunk.Reasoning.Summary)
			}
			if chunk.Reasoning.Content != "" {
				if reasoningContent.Len() > 0 {
					reasoningContent.WriteString("\n")
				}
				reasoningContent.WriteString(chunk.Reasoning.Content)
			}
		}

		if chunk.ToolCallDelta != nil {
			key := chunk.ToolCallDelta.ID
			if key == "" {
				key = fmt.Sprintf("index:%d", chunk.ToolCallDelta.Index)
			}

			toolCall := toolCallsByIdentifier[key]
			if toolCall == nil {
				toolCall = &ToolCall{
					ID:   chunk.ToolCallDelta.ID,
					Type: chunk.ToolCallDelta.Type,
					Function: FunctionCall{
						Name: chunk.ToolCallDelta.FunctionName,
					},
				}
				if toolCall.Type == "" {
					toolCall.Type = "function"
				}
				toolCallsByIdentifier[key] = toolCall
				toolCallOrder = append(toolCallOrder, key)
			}

			if chunk.ToolCallDelta.FunctionName != "" {
				toolCall.Function.Name = chunk.ToolCallDelta.FunctionName
			}
			if chunk.ToolCallDelta.ArgumentsDelta != "" {
				toolCall.Function.Arguments += chunk.ToolCallDelta.ArgumentsDelta
			}
		}

		if chunk.Usage != nil {
			resp.Usage = *chunk.Usage
		}

		if chunk.Done {
			break
		}
	}

	if len(contentByOutputIndex) > 0 || len(refusalByOutputIndex) > 0 {
		indices := make([]int, 0, len(contentByOutputIndex)+len(refusalByOutputIndex))
		for index := range contentByOutputIndex {
			indices = append(indices, index)
		}
		for index := range refusalByOutputIndex {
			indices = append(indices, index)
		}
		sort.Ints(indices)
		var merged strings.Builder
		seen := map[int]bool{}
		for _, index := range indices {
			if seen[index] {
				continue
			}
			seen[index] = true
			if builder := contentByOutputIndex[index]; builder != nil && builder.Len() > 0 {
				merged.WriteString(builder.String())
				continue
			}
			if builder := refusalByOutputIndex[index]; builder != nil {
				merged.WriteString(builder.String())
			}
		}
		merged.WriteString(contentBuilder.String())
		resp.Content = merged.String()
	} else {
		resp.Content = contentBuilder.String()
	}
	if resp.Content == "" && refusalBuilder.Len() > 0 {
		resp.Content = refusalBuilder.String()
	}

	if reasoningSummary.Len() > 0 || reasoningContent.Len() > 0 {
		resp.Reasoning = &ReasoningData{
			Summary: reasoningSummary.String(),
			Content: reasoningContent.String(),
		}
	}

	if len(toolCallOrder) > 0 {
		resp.ToolCalls = make([]ToolCall, 0, len(toolCallOrder))
		for _, key := range toolCallOrder {
			if toolCall := toolCallsByIdentifier[key]; toolCall != nil {
				resp.ToolCalls = append(resp.ToolCalls, *toolCall)
			}
		}
	}

	return &resp, nil
}
