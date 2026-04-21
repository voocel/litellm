package providers

import "testing"

// decodeGeminiToolResponse must accept non-JSON content (e.g. the synthetic
// plain-text orphan compensation from PrepareMessages) without erroring, and
// route errors under "error" key rather than "result". A single table test
// covers every branch.
func TestDecodeGeminiToolResponse(t *testing.T) {
	cases := []struct {
		name    string
		msg     Message
		wantKey string // key that must be present in the map
	}{
		{"valid JSON object preserved", Message{Content: `{"temp":72}`}, "temp"},
		{"non-JSON wraps under result", Message{Content: "72°F"}, "result"},
		{"IsError routes under error", Message{Content: "boom", IsError: true}, "error"},
		{"empty + IsError synthesizes error", Message{IsError: true}, "error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := decodeGeminiToolResponse(tc.msg)
			if _, ok := got[tc.wantKey]; !ok {
				t.Fatalf("missing key %q in %v", tc.wantKey, got)
			}
		})
	}
}

// Core Gemini contract per https://ai.google.dev/gemini-api/docs/function-calling :
//   - functionCall and functionResponse both echo the official id (Gemini 3+)
//   - functionResponse.name comes from the matching assistant tool_call
//   - role mapping: assistant → model, tool → user
//
// Single scenario walks a full user → model → tool roundtrip.
func TestGeminiBuildContentsToolRoundtrip(t *testing.T) {
	p := &GeminiProvider{}
	req := &Request{
		Messages: []Message{
			{Role: "system", Content: "be concise"},
			{Role: "user", Content: "weather in Paris?"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{
					{ID: "call_xyz", Type: "function", Function: FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
				},
			},
			{Role: "tool", ToolCallID: "call_xyz", Content: `{"temp":"15C"}`},
		},
	}
	contents, sys, err := p.buildContents(req)
	if err != nil {
		t.Fatalf("buildContents failed: %v", err)
	}
	if sys != "be concise" {
		t.Fatalf("system message not extracted: %q", sys)
	}
	if len(contents) != 3 {
		t.Fatalf("want 3 Content blocks (user, model, user/tool), got %d", len(contents))
	}

	model := contents[1]
	if model.Role != "model" || model.Parts[0].FunctionCall == nil {
		t.Fatalf("assistant must map to role=model with functionCall part: %+v", model)
	}
	if model.Parts[0].FunctionCall.ID != "call_xyz" {
		t.Fatalf("functionCall.id must echo tool_call.ID, got %q", model.Parts[0].FunctionCall.ID)
	}

	tool := contents[2]
	if tool.Role != "user" || tool.Parts[0].FunctionResponse == nil {
		t.Fatalf("tool result must map to role=user with functionResponse part: %+v", tool)
	}
	fr := tool.Parts[0].FunctionResponse
	if fr.ID != "call_xyz" {
		t.Fatalf("functionResponse.id must echo tool_call.ID, got %q", fr.ID)
	}
	if fr.Name != "get_weather" {
		t.Fatalf("functionResponse.name must come from matching tool_call, got %q", fr.Name)
	}
}

func TestGeminiStreamReaderReusesIndexForSameFunctionCallID(t *testing.T) {
	r := &geminiStreamReader{
		provider:        "gemini",
		model:           "gemini-test",
		toolCallIndexes: make(map[string]int),
	}

	first, err := r.processResponse(geminiStreamResponse{
		Candidates: []geminiCandidate{{
			Content: geminiContent{
				Parts: []geminiPart{{
					FunctionCall: &geminiFunctionCall{
						ID:   "call_same",
						Name: "lookup",
						Args: map[string]any{"a": 1},
					},
				}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("first processResponse failed: %v", err)
	}

	second, err := r.processResponse(geminiStreamResponse{
		Candidates: []geminiCandidate{{
			Content: geminiContent{
				Parts: []geminiPart{{
					FunctionCall: &geminiFunctionCall{
						ID:   "call_same",
						Name: "lookup",
						Args: map[string]any{"b": 2},
					},
				}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("second processResponse failed: %v", err)
	}

	if first == nil || first.ToolCallDelta == nil || second == nil || second.ToolCallDelta == nil {
		t.Fatalf("expected tool call deltas, got first=%+v second=%+v", first, second)
	}
	if first.ToolCallDelta.ID != "call_same" || second.ToolCallDelta.ID != "call_same" {
		t.Fatalf("tool call ids not preserved: first=%+v second=%+v", first.ToolCallDelta, second.ToolCallDelta)
	}
	if first.ToolCallDelta.Index != second.ToolCallDelta.Index {
		t.Fatalf("same function call id must reuse stream index: first=%d second=%d", first.ToolCallDelta.Index, second.ToolCallDelta.Index)
	}
}
