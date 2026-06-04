package providers

import (
	"bufio"
	"encoding/json"
	"strings"
	"testing"
)

// Core OpenAI Responses API contract per
// https://platform.openai.com/docs/api-reference/responses :
//   - user text      → {type: message, role: user, content: [input_text]}
//   - assistant text → {type: message, role: assistant, content: [output_text]}
//   - tool_calls     → {type: function_call, call_id, name, arguments}
//   - tool result    → {type: function_call_output, call_id, output}
//   - role=tool never appears as a message role
//
// A single full conversation exercises every item type and their ordering.
func TestResponsesInputConvertsAllItemTypes(t *testing.T) {
	items := convertMessagesToResponsesInput([]Message{
		{Role: "user", Content: "weather?"},
		{
			Role:    "assistant",
			Content: "let me check",
			ToolCalls: []ToolCall{
				{ID: "call_1", Type: "function", Function: FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
			},
		},
		{Role: "tool", ToolCallID: "call_1", Content: `{"temp":"15C"}`},
	})

	if len(items) != 4 {
		t.Fatalf("want 4 items (user msg, assistant msg, function_call, function_call_output), got %d: %+v", len(items), items)
	}

	// user
	if items[0].Type != "message" || items[0].Role != "user" ||
		len(items[0].Content) != 1 || items[0].Content[0].Type != "input_text" {
		t.Fatalf("user item malformed: %+v", items[0])
	}
	// assistant text (must precede function_call items — mirrors model output order)
	if items[1].Type != "message" || items[1].Role != "assistant" ||
		len(items[1].Content) != 1 || items[1].Content[0].Type != "output_text" {
		t.Fatalf("assistant message item malformed: %+v", items[1])
	}
	// function_call
	fc := items[2]
	if fc.Type != "function_call" || fc.CallID != "call_1" || fc.Name != "get_weather" || fc.Arguments == "" {
		t.Fatalf("function_call item malformed: %+v", fc)
	}
	if fc.Role != "" || len(fc.Content) != 0 {
		t.Fatalf("function_call must not carry role/content: %+v", fc)
	}
	// function_call_output
	fo := items[3]
	if fo.Type != "function_call_output" || fo.CallID != "call_1" || fo.Output == "" {
		t.Fatalf("function_call_output item malformed: %+v", fo)
	}
	if fo.Role == "tool" {
		t.Fatalf("role=tool must never leak into input items")
	}
}

func TestResponsesToolsNilStrictUsesResponsesStrictDefault(t *testing.T) {
	tools, err := convertResponsesAPITools([]Tool{{
		Type: "function",
		Function: FunctionDef{
			Name: "get_weather",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []any{"city"},
			},
		},
	}})
	if err != nil {
		t.Fatalf("convertResponsesAPITools: %v", err)
	}
	if tools[0].Strict != nil {
		t.Fatalf("strict should be omitted so Responses API default applies: %v", tools[0].Strict)
	}
	params := tools[0].Parameters.(map[string]any)
	if params["additionalProperties"] != false {
		t.Fatalf("nil strict should still normalise schema for Responses default strict=true: %v", params)
	}
}

func TestResponsesBuildRequestMapsGPT55OfficialFields(t *testing.T) {
	p := NewOpenAI(ProviderConfig{APIKey: "test"})
	maxOutputTokens := 1200
	maxToolCalls := 4
	topLogprobs := 2
	parallelToolCalls := true
	store := true
	background := false

	req, err := p.buildResponsesAPIRequest(&OpenAIResponsesRequest{
		Model: "gpt-5.5",
		Messages: []Message{
			{Role: "developer", Content: "Follow the contract."},
			{Role: "user", Content: "Search and summarize."},
		},
		Instructions:         "Be concise.",
		PreviousResponseID:   "resp_previous",
		MaxOutputTokens:      &maxOutputTokens,
		MaxToolCalls:         &maxToolCalls,
		Include:              []string{"reasoning.encrypted_content"},
		TopLogprobs:          &topLogprobs,
		TextVerbosity:        "low",
		Truncation:           "auto",
		OpenAITools:          []OpenAIResponsesTool{{"type": "web_search_preview"}},
		ParallelToolCalls:    &parallelToolCalls,
		ReasoningEffort:      "xhigh",
		ReasoningSummary:     "auto",
		PromptCacheKey:       "workflow-v1",
		PromptCacheRetention: "24h",
		Metadata:             map[string]string{"tenant": "acme"},
		SafetyIdentifier:     "user-123",
		ServiceTier:          "flex",
		Store:                &store,
		Background:           &background,
		Prompt:               map[string]any{"id": "pmpt_123"},
	})
	if err != nil {
		t.Fatalf("buildResponsesAPIRequest: %v", err)
	}

	if req.Model != "gpt-5.5" {
		t.Fatalf("model = %q, want gpt-5.5", req.Model)
	}
	if req.Instructions != "Be concise.\nFollow the contract." {
		t.Fatalf("instructions = %q", req.Instructions)
	}
	if req.PreviousResponseID != "resp_previous" {
		t.Fatalf("previous_response_id = %q", req.PreviousResponseID)
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != maxOutputTokens {
		t.Fatalf("max_output_tokens = %v", req.MaxOutputTokens)
	}
	if req.MaxToolCalls == nil || *req.MaxToolCalls != maxToolCalls {
		t.Fatalf("max_tool_calls = %v", req.MaxToolCalls)
	}
	if len(req.Include) != 1 || req.Include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v", req.Include)
	}
	if req.TopLogprobs == nil || *req.TopLogprobs != topLogprobs {
		t.Fatalf("top_logprobs = %v", req.TopLogprobs)
	}
	if req.Text == nil || req.Text.Verbosity != "low" {
		t.Fatalf("text verbosity = %#v", req.Text)
	}
	if req.Truncation != "auto" {
		t.Fatalf("truncation = %q", req.Truncation)
	}
	if req.ParallelToolCalls == nil || !*req.ParallelToolCalls {
		t.Fatalf("parallel_tool_calls = %v", req.ParallelToolCalls)
	}
	if req.Reasoning == nil || req.Reasoning.Effort != "xhigh" || req.Reasoning.Summary != "auto" {
		t.Fatalf("reasoning = %#v", req.Reasoning)
	}
	if req.PromptCacheKey != "workflow-v1" || req.PromptCacheRetention != "24h" {
		t.Fatalf("prompt cache fields not mapped: %#v", req)
	}
	if req.Metadata["tenant"] != "acme" || req.SafetyIdentifier != "user-123" {
		t.Fatalf("metadata/safety fields not mapped: %#v", req)
	}
	if req.ServiceTier != "flex" || req.Store == nil || !*req.Store || req.Background == nil || *req.Background {
		t.Fatalf("service fields not mapped: %#v", req)
	}
	if req.Prompt["id"] != "pmpt_123" {
		t.Fatalf("prompt = %#v", req.Prompt)
	}

	toolsBody, err := json.Marshal(req.Tools)
	if err != nil {
		t.Fatalf("marshal tools: %v", err)
	}
	if string(toolsBody) != `[{"type":"web_search_preview"}]` {
		t.Fatalf("tools json = %s", toolsBody)
	}
}

func TestResponsesBuildRequestUsesThinkingLevelAsReasoningEffort(t *testing.T) {
	p := NewOpenAI(ProviderConfig{APIKey: "test"})
	req, err := p.buildResponsesAPIRequest(&OpenAIResponsesRequest{
		Model:    "gpt-5.5",
		Messages: []Message{{Role: "user", Content: "solve"}},
		Thinking: &ThinkingConfig{Type: "enabled", Level: "high"},
	})
	if err != nil {
		t.Fatalf("buildResponsesAPIRequest: %v", err)
	}
	if req.Reasoning == nil || req.Reasoning.Effort != "high" {
		t.Fatalf("reasoning = %#v, want effort=high", req.Reasoning)
	}
}

func TestResponsesOutputExtractsOfficialFunctionCall(t *testing.T) {
	_, _, toolCalls, _ := extractResponsesOutputItems([]responsesAPIOutputItem{{
		Type:      "function_call",
		ID:        "fc_1",
		CallID:    "call_1",
		Name:      "get_weather",
		Arguments: `{"city":"Boston"}`,
	}}, "")

	if len(toolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].ID != "call_1" || toolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool call malformed: %#v", toolCalls[0])
	}
}

func TestResponsesRejectsConversationWithPreviousResponseID(t *testing.T) {
	p := NewOpenAI(ProviderConfig{APIKey: "test"})
	_, err := p.buildResponsesAPIRequest(&OpenAIResponsesRequest{
		Model:              "gpt-5.5",
		Messages:           []Message{{Role: "user", Content: "hello"}},
		Conversation:       "conv_123",
		PreviousResponseID: "resp_123",
	})
	if err == nil {
		t.Fatal("expected mutual exclusion error")
	}
}

// Regression: OpenAI's Responses API (gpt-5*) delivers a function call's NAME
// only in response.output_item.added; the argument-delta events carry args but
// no name. The reader must forward that name as a tool_call_delta at call start
// (mirroring the chat-completions stream), so downstream accumulation captures
// it. Otherwise the assembled tool call has full args and an empty name, and
// the host dispatches `tool "" not found` / OpenAI later rejects the history
// with "function name is required".
func TestResponsesStreamForwardsFunctionNameAtCallStart(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","name":"get_weather","call_id":"call_1","arguments":""},"sequence_number":1}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"city\":","sequence_number":2}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"\"SF\"}","sequence_number":3}`,
		``,
		`event: response.function_call_arguments.done`,
		`data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"name":"get_weather","arguments":"{\"city\":\"SF\"}","sequence_number":4}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","sequence_number":5,"response":{"id":"resp_1","status":"completed"}}`,
		``,
	}, "\n")

	r := &responsesAPIStreamReader{
		scanner:         bufio.NewScanner(strings.NewReader(sse)),
		provider:        "openai",
		model:           "gpt-5.4-mini",
		toolCallSeenMap: make(map[string]bool),
	}

	var startName, args string
	for {
		c, err := r.Next()
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if c.Done {
			break
		}
		if c.ToolCallDelta != nil {
			if c.Type == "tool_call_delta" && c.ToolCallDelta.FunctionName != "" && startName == "" {
				startName = c.ToolCallDelta.FunctionName
			}
			args += c.ToolCallDelta.ArgumentsDelta
		}
	}

	if startName != "get_weather" {
		t.Fatalf("expected a tool_call_delta carrying the function name at call start, got %q (the Responses API only sends the name in output_item.added; it must be forwarded so downstream captures it)", startName)
	}
	if args != `{"city":"SF"}` {
		t.Fatalf("arguments = %q, want the accumulated JSON", args)
	}
}

// Regression: the stream accumulator keys tool calls by ToolCallDelta.Index.
// The Responses API carries the slot in output_index, so the reader must map
// output_index -> Index on every tool-call event; otherwise parallel function
// calls all default to index 0 and get merged. (voocel/litellm#4 review.)
func TestResponsesStreamMapsOutputIndexToIndex(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","name":"tool_a","call_id":"call_0","arguments":""},"sequence_number":1}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","name":"tool_b","call_id":"call_1","arguments":""},"sequence_number":2}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{}","sequence_number":3}`,
		``,
		`event: response.function_call_arguments.done`,
		`data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":1,"name":"tool_b","arguments":"{}","sequence_number":4}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","sequence_number":5,"response":{"id":"resp_1","status":"completed"}}`,
		``,
	}, "\n")

	r := &responsesAPIStreamReader{
		scanner:         bufio.NewScanner(strings.NewReader(sse)),
		provider:        "openai",
		model:           "gpt-5.4-mini",
		toolCallSeenMap: make(map[string]bool),
	}

	// Map each tool call's item id to the Index values its events carried.
	idxByItem := map[string]int{}
	for {
		c, err := r.Next()
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if c.Done {
			break
		}
		d := c.ToolCallDelta
		if d == nil || d.ItemID == "" {
			continue
		}
		if prev, ok := idxByItem[d.ItemID]; ok && prev != d.Index {
			t.Fatalf("item %s saw inconsistent Index %d vs %d", d.ItemID, prev, d.Index)
		}
		idxByItem[d.ItemID] = d.Index
	}

	if idxByItem["fc_0"] != 0 {
		t.Fatalf("fc_0 Index = %d, want 0", idxByItem["fc_0"])
	}
	if idxByItem["fc_1"] != 1 {
		t.Fatalf("fc_1 Index = %d, want 1 (output_index must map to Index so the accumulator keeps parallel calls separate)", idxByItem["fc_1"])
	}
}
