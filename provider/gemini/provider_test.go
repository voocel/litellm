package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/retry"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestNewRejectsAmbiguousTransportConfig(t *testing.T) {
	_, err := New(Config{
		APIKey:     "test-key",
		HTTPClient: roundTripFunc(nil),
		Transport:  roundTripperFunc(nil),
	})
	if err == nil || !strings.Contains(err.Error(), "HTTPClient and Transport are mutually exclusive") {
		t.Fatalf("expected HTTPClient/Transport error, got %v", err)
	}

	_, err = New(Config{
		APIKey:     "test-key",
		HTTPClient: roundTripFunc(nil),
		Retry:      retry.DefaultPolicy(),
	})
	if err == nil || !strings.Contains(err.Error(), "Retry cannot be used with a custom HTTPClient") {
		t.Fatalf("expected HTTPClient/Retry error, got %v", err)
	}
}

func TestBuildRequestToolRoundTripPreservesSignatureAndName(t *testing.T) {
	provider := mustProvider(t)
	req := &litellm.Request{
		Model: "gemini-3-pro",
		Messages: []litellm.Message{
			litellm.System("be concise"),
			litellm.User(
				litellm.Text("weather?"),
				litellm.ImageBlock{FileURI: "gs://bucket/image.png", MIME: "image/png"},
			),
			litellm.Assistant(
				litellm.ReasoningBlock{Text: "Need weather.", Signature: "sig-think"},
				litellm.ToolUseBlock{ID: "call_weather", Name: "get_weather", Arguments: litellm.MustJSONRaw(map[string]any{"city": "Paris"}), Signature: "sig-call"},
			),
			litellm.ToolResultText("call_weather", `{"temp":"15C"}`),
		},
		Tools: []litellm.Tool{
			mustTool(t, "get_weather", "Get weather.", map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			}),
		},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "low", IncludeOutput: true},
	}
	wire, err := provider.buildRequest(req)
	if err != nil {
		t.Fatalf("buildRequest returned error: %v", err)
	}
	testgolden.AssertJSON(t, "../../testdata/gemini/request_multimodal_tools.golden.json", wire)

	data, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal wire: %v", err)
	}
	jsonText := string(data)
	for _, want := range []string{
		`"systemInstruction"`,
		`"fileUri":"gs://bucket/image.png"`,
		`"thought":true`,
		`"thoughtSignature":"sig-think"`,
		`"thoughtSignature":"sig-call"`,
		`"functionCall"`,
		`"id":"call_weather"`,
		`"functionResponse"`,
		`"name":"get_weather"`,
		`"thinkingLevel":"low"`,
	} {
		if !strings.Contains(jsonText, want) {
			t.Fatalf("wire JSON missing %s:\n%s", want, jsonText)
		}
	}
	if strings.Index(jsonText, `"functionCall"`) > strings.Index(jsonText, `"functionResponse"`) {
		t.Fatalf("functionCall must precede functionResponse:\n%s", jsonText)
	}
}

func TestBuildRequestGemini3AddsPlaceholderForFirstUnsignedToolUse(t *testing.T) {
	provider := mustProvider(t)
	wire, err := provider.buildRequest(&litellm.Request{
		Model: "gemini-3-pro",
		Messages: []litellm.Message{
			litellm.Assistant(
				litellm.ToolUseBlock{ID: "call_1", Name: "a", Arguments: litellm.MustJSONRaw(map[string]any{})},
				litellm.ToolUseBlock{ID: "call_2", Name: "b", Arguments: litellm.MustJSONRaw(map[string]any{})},
			),
		},
	})
	if err != nil {
		t.Fatalf("buildRequest returned error: %v", err)
	}
	parts := wire.Contents[0].Parts
	if parts[0].ThoughtSignature != thoughtSignaturePlaceholder {
		t.Fatalf("first signature = %q, want placeholder", parts[0].ThoughtSignature)
	}
	if parts[1].ThoughtSignature != "" {
		t.Fatalf("parallel call signature = %q, want empty", parts[1].ThoughtSignature)
	}
}

func TestBuildRequestRoundTripsResponseBlocksWithSignatures(t *testing.T) {
	provider := mustProvider(t)
	prev := &litellm.Response{
		Blocks: []litellm.Block{
			litellm.ReasoningBlock{Text: "Need lookup.", Signature: "sig-think"},
			litellm.ToolUseBlock{
				ID:        "call_1",
				Name:      "lookup",
				Arguments: litellm.MustJSONRaw(map[string]any{"q": "x"}),
				Signature: "sig-call",
			},
		},
	}
	wire, err := provider.buildRequest(&litellm.Request{
		Model: "gemini-3-pro",
		Messages: []litellm.Message{
			litellm.Assistant(prev.Blocks...),
			litellm.ToolResultText("call_1", `{"ok":true}`),
		},
	})
	if err != nil {
		t.Fatalf("buildRequest returned error: %v", err)
	}
	assistant := wire.Contents[0].Parts
	if len(assistant) != 2 {
		t.Fatalf("assistant parts = %#v", assistant)
	}
	if assistant[0].ThoughtSignature != "sig-think" || assistant[1].ThoughtSignature != "sig-call" {
		t.Fatalf("signatures = %q/%q", assistant[0].ThoughtSignature, assistant[1].ThoughtSignature)
	}
	result := wire.Contents[1].Parts[0].FunctionResponse
	if result == nil || result.Name != "lookup" || result.ID != "call_1" {
		t.Fatalf("function response = %#v", result)
	}
}

func TestBuildRequestRejectsThinkingBudget(t *testing.T) {
	provider := mustProvider(t)
	budget := 8192
	_, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-3.7-flash",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, BudgetTokens: &budget},
	})
	if err == nil || !strings.Contains(err.Error(), "budget_tokens is not supported") {
		t.Fatalf("expected unsupported budget error, got %v", err)
	}
}

func TestBuildRequestGemini3AndLaterUseThinkingLevel(t *testing.T) {
	provider := mustProvider(t)
	wire, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-4-flash",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "high"},
	})
	if err != nil {
		t.Fatalf("buildRequest returned error: %v", err)
	}
	tc := wire.GenerationConfig.ThinkingConfig
	if tc == nil || tc.ThinkingLevel != "high" {
		t.Fatalf("thinking config = %+v, want effort high", tc)
	}
}

func TestThinkingCapabilitiesFollowAPIGeneration(t *testing.T) {
	provider := mustProvider(t)
	caps := provider.Capabilities("models/gemini-4-flash")
	if caps.Thinking.Supported != litellm.SupportYes || caps.Thinking.Disable != litellm.SupportNo || !caps.Thinking.SupportsEffort("high") || caps.Thinking.SupportsEffort("minimal") || caps.Thinking.SupportsEffort("medium") {
		t.Fatalf("future Gemini capabilities = %+v", caps.Thinking)
	}
	if legacy := provider.Capabilities("gemini-2.5-flash").Thinking; legacy.Supported != litellm.SupportUnknown {
		t.Fatalf("legacy Gemini capabilities = %+v", legacy)
	}
}

func TestBuildRequestRejectsLegacyThinkingModel(t *testing.T) {
	provider := mustProvider(t)
	_, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-2.5-flash",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled, Effort: "high"},
	})
	if err == nil || !strings.Contains(err.Error(), "thinking is not supported") {
		t.Fatalf("expected unsupported model error, got %v", err)
	}
}

func TestBuildRequestRejectsDisabledThinking(t *testing.T) {
	provider := mustProvider(t)
	_, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-3.6-flash",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingDisabled},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot be disabled") {
		t.Fatalf("expected disable error, got %v", err)
	}
}

func TestBuildRequestRejectsToolResultWithoutPrecedingToolUse(t *testing.T) {
	provider := mustProvider(t)
	_, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.ToolResultText("missing", "result")},
	})
	if err == nil || !strings.Contains(err.Error(), "no preceding tool use name") {
		t.Fatalf("expected tool result mapping error, got %v", err)
	}
}

func TestBuildRequestPreservesNullableToolJSONSchema(t *testing.T) {
	provider := mustProvider(t)
	schema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"issues": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {
						"suggestion": {"type": ["string", "null"]}
					},
					"required": ["suggestion"],
					"additionalProperties": false
				}
			}
		},
		"required": ["issues"],
		"additionalProperties": false
	}`)
	var want map[string]any
	if err := json.Unmarshal(schema, &want); err != nil {
		t.Fatal(err)
	}
	for _, strict := range []litellm.StrictMode{litellm.StrictDefault, litellm.StrictEnabled, litellm.StrictDisabled} {
		t.Run(fmt.Sprintf("strict=%d", strict), func(t *testing.T) {
			tool := mustTool(t, "audit_foundation", "Audit foundation.", schema)
			tool.Strict = strict
			wire, err := provider.buildRequest(&litellm.Request{
				Model:      "gemini-3-pro",
				Messages:   []litellm.Message{litellm.UserText("Audit the foundation.")},
				Tools:      []litellm.Tool{tool},
				ToolChoice: "auto",
			})
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Tools []struct {
					FunctionDeclarations []map[string]any `json:"functionDeclarations"`
				} `json:"tools"`
			}
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			declaration := body.Tools[0].FunctionDeclarations[0]
			if _, ok := declaration["parameters"]; ok {
				t.Error("JSON Schema must not be sent through OpenAPI parameters")
			}
			if got := declaration["parametersJsonSchema"]; !reflect.DeepEqual(got, want) {
				t.Errorf("parametersJsonSchema = %#v, want %#v", got, want)
			}
		})
	}
}

func TestBuildRequestStrictToolChoice(t *testing.T) {
	provider := mustProvider(t)
	tool := mustTool(t, "lookup", "Lookup.", map[string]any{"type": "object"})
	tool.Strict = litellm.StrictEnabled
	for _, test := range []struct {
		name   string
		choice any
		mode   string
	}{
		{name: "default", mode: "VALIDATED"},
		{name: "auto", choice: "auto", mode: "VALIDATED"},
		{name: "required", choice: "required", mode: "ANY"},
		{name: "none", choice: "none", mode: "NONE"},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire, err := provider.buildRequest(&litellm.Request{
				Model:      "gemini-3-pro",
				Messages:   []litellm.Message{litellm.UserText("hi")},
				Tools:      []litellm.Tool{tool},
				ToolChoice: test.choice,
			})
			if err != nil {
				t.Fatalf("buildRequest returned error: %v", err)
			}
			if got := wire.ToolConfig.FunctionCallingConfig.Mode; got != test.mode {
				t.Fatalf("mode = %q, want %q", got, test.mode)
			}
		})
	}
}

func TestBuildRequestRejectsInvalidToolChoice(t *testing.T) {
	provider := mustProvider(t)
	tool := mustTool(t, "lookup", "Lookup.", map[string]any{"type": "object"})
	tool.Strict = litellm.StrictEnabled
	_, err := provider.buildRequest(&litellm.Request{
		Model:      "gemini-3-pro",
		Messages:   []litellm.Message{litellm.UserText("hi")},
		Tools:      []litellm.Tool{tool},
		ToolChoice: "invalid",
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported tool choice") {
		t.Fatalf("expected tool choice error, got %v", err)
	}
}

func TestBuildRequestMixedStrictTools(t *testing.T) {
	provider := mustProvider(t)
	strict := mustTool(t, "strict", "Strict.", map[string]any{"type": "object"})
	strict.Strict = litellm.StrictEnabled
	other := mustTool(t, "other", "Other.", map[string]any{"type": "object"})

	base := litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Tools:    []litellm.Tool{strict, other},
	}
	wire, err := provider.buildRequest(&base)
	if err != nil {
		t.Fatalf("default strict mode returned error: %v", err)
	}
	if got := wire.ToolConfig.FunctionCallingConfig.Mode; got != "VALIDATED" {
		t.Fatalf("mode = %q, want VALIDATED", got)
	}

	base.Tools[1].Strict = litellm.StrictDisabled
	_, err = provider.buildRequest(&base)
	if err == nil || !strings.Contains(err.Error(), "enabled and disabled") {
		t.Fatalf("expected conflicting strict mode error, got %v", err)
	}
}

func TestBuildRequestProviderOptions(t *testing.T) {
	provider := mustProvider(t)
	wire, err := provider.buildRequest(&litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: litellm.ProviderOptions{
			ProviderOptionTopK:           10,
			ProviderOptionCandidateCount: 1,
			ProviderOptionSafetySettings: []map[string]any{
				{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
			},
		},
	})
	if err != nil {
		t.Fatalf("buildRequest returned error: %v", err)
	}
	if wire.GenerationConfig == nil || wire.GenerationConfig.TopK == nil || *wire.GenerationConfig.TopK != 10 {
		t.Fatalf("generation config topK = %+v", wire.GenerationConfig)
	}
	if wire.GenerationConfig.CandidateCount == nil || *wire.GenerationConfig.CandidateCount != 1 {
		t.Fatalf("generation config candidateCount = %+v", wire.GenerationConfig)
	}
	if len(wire.SafetySettings) != 1 || wire.SafetySettings[0].Category != "HARM_CATEGORY_DANGEROUS_CONTENT" || wire.SafetySettings[0].Threshold != "BLOCK_ONLY_HIGH" {
		t.Fatalf("safety settings = %+v", wire.SafetySettings)
	}
}

func TestBuildRequestRejectsUnknownProviderOption(t *testing.T) {
	provider := mustProvider(t)
	_, err := provider.buildRequest(&litellm.Request{
		Model:           "gemini-3-pro",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: litellm.ProviderOptions{"unknown": true},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported provider option") {
		t.Fatalf("expected provider option error, got %v", err)
	}
}

func TestChatConvertsThinkingToolAndUsage(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.RawQuery != "" {
				t.Fatalf("request URL contains query credentials: %s", req.URL.String())
			}
			if got := req.Header.Get("x-goog-api-key"); got != "test-key" {
				t.Fatalf("x-goog-api-key = %q, want test-key", got)
			}
			return jsonResponse(200, `{
				"candidates":[{
					"content":{"parts":[
						{"text":"thinking","thought":true,"thoughtSignature":"sig-think"},
						{"text":"answer"},
						{"thoughtSignature":"sig-call","functionCall":{"id":"call_1","name":"lookup","args":{"q":"x"}}}
					]},
					"finishReason":"FUNCTION_CALLING"
				}],
				"usageMetadata":{
					"promptTokenCount":3,
					"candidatesTokenCount":4,
					"thoughtsTokenCount":2,
					"totalTokenCount":9,
					"cachedContentTokenCount":1
				}
			}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := provider.Chat(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "answer" || resp.Reasoning() != "thinking" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || calls[0].Signature != "sig-call" || string(calls[0].Arguments) != `{"q":"x"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if resp.FinishReason != litellm.FinishReasonToolCall {
		t.Fatalf("finish reason = %q", resp.FinishReason)
	}
	if resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 4 || resp.Usage.ReasoningTokens != 2 || resp.Usage.CacheReadTokens != 1 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestConvertResponseRejectsNil(t *testing.T) {
	_, err := convertResponse(nil, &litellm.Request{Model: "gemini-3-pro"})
	if err == nil || !strings.Contains(err.Error(), "response cannot be nil") {
		t.Fatalf("expected nil response error, got %v", err)
	}
}

func TestNetworkErrorDoesNotExposeAPIKey(t *testing.T) {
	const secret = "gemini-secret-key"
	provider, err := New(Config{
		APIKey:  secret,
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("request to %s failed", req.URL.String())
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil {
		t.Fatal("expected network error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("network error exposed API key: %v", err)
	}
}

func TestChatWarnsWhenFunctionCallIDIsSynthesized(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(200, `{
				"candidates":[{
					"content":{"parts":[
						{"functionCall":{"name":"lookup","args":{"q":"x"}}}
					]}
				}]
			}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := provider.Chat(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || !strings.HasPrefix(calls[0].ID, "call_") {
		t.Fatalf("tool calls = %+v", calls)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "gemini.tool_call_id_synthesized" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

func TestChatPromptFeedbackReturnsProviderError(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(200, `{
				"promptFeedback":{
					"blockReason":"SAFETY",
					"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]
				}
			}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil {
		t.Fatal("expected prompt feedback error")
	}
	if !strings.Contains(err.Error(), "prompt blocked: SAFETY") || !strings.Contains(err.Error(), "HARM_CATEGORY_DANGEROUS_CONTENT=HIGH,blocked") {
		t.Fatalf("error did not expose prompt feedback: %v", err)
	}
}

func TestChatCandidateSafetyFinishReturnsProviderError(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(200, `{
				"candidates":[{
					"finishReason":"SAFETY",
					"finishMessage":"blocked",
					"safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]
				}]
			}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil {
		t.Fatal("expected safety finish error")
	}
	if !strings.Contains(err.Error(), "finish_reason=SAFETY") || !strings.Contains(err.Error(), "blocked") {
		t.Fatalf("error did not expose safety finish: %v", err)
	}
}

func TestStreamConvertsSSEToTypedEvents(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.Header.Get("Accept") != "text/event-stream" {
				t.Fatalf("Accept = %q, want text/event-stream", req.Header.Get("Accept"))
			}
			if req.URL.Query().Get("alt") != "sse" || req.URL.Query().Has("key") {
				t.Fatalf("stream URL query = %q, want only alt=sse", req.URL.RawQuery)
			}
			if got := req.Header.Get("x-goog-api-key"); got != "test-key" {
				t.Fatalf("x-goog-api-key = %q, want test-key", got)
			}
			return streamResponse(testgolden.ReadFixtureString(t, "../../testdata/gemini/stream.jsonl")), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "answer" || resp.Reasoning() != "think" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	reasoning, ok := resp.Blocks[0].(litellm.ReasoningBlock)
	if !ok || reasoning.Signature != "sig-think" {
		t.Fatalf("reasoning block = %+v", resp.Blocks[0])
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || calls[0].Signature != "sig-call" || string(calls[0].Arguments) != `{"q":"x"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 4 || resp.Usage.ReasoningTokens != 2 || resp.Usage.CacheReadTokens != 1 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.FinishReason != litellm.FinishReasonToolCall {
		t.Fatalf("finish reason = %q", resp.FinishReason)
	}
}

func TestListModelsUsesHeaderAuthentication(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.RawQuery != "" {
				t.Fatalf("models URL contains query credentials: %s", req.URL.String())
			}
			if got := req.Header.Get("x-goog-api-key"); got != "test-key" {
				t.Fatalf("x-goog-api-key = %q, want test-key", got)
			}
			return jsonResponse(http.StatusOK, `{"models":[]}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels returned error: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("models = %+v, want none", models)
	}
}

func TestStreamWarnsWhenFunctionCallIDIsSynthesized(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(strings.Join([]string{
				`data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"x"}}}]},"finishReason":"FUNCTION_CALLING"}]}`,
				`data: [DONE]`,
				``,
			}, "\n")), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || !strings.HasPrefix(calls[0].ID, "call_") {
		t.Fatalf("tool calls = %+v", calls)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "gemini.tool_call_id_synthesized" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

func TestStreamParsesJSONArrayChunks(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(`[{"candidates":[{"content":{"parts":[{"text":"a"}]}}]},{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}]}]`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "ab" || resp.FinishReason != litellm.FinishReasonStop {
		t.Fatalf("response text/finish = %q/%q", resp.Text(), resp.FinishReason)
	}
}

func TestStreamPromptFeedbackReturnsProviderError(t *testing.T) {
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://example.test",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(`data: {"promptFeedback":{"blockReason":"SAFETY","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH","blocked":true}]}}`), nil
		}),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "gemini-3-pro",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	_, err = litellm.Collect(stream)
	if err == nil {
		t.Fatal("expected prompt feedback error")
	}
	if !strings.Contains(err.Error(), "prompt blocked: SAFETY") || !strings.Contains(err.Error(), "HARM_CATEGORY_DANGEROUS_CONTENT=HIGH,blocked") {
		t.Fatalf("error did not expose prompt feedback: %v", err)
	}
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func streamResponse(body string) *http.Response {
	resp := jsonResponse(http.StatusOK, body)
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp
}

func mustProvider(t *testing.T) *Provider {
	t.Helper()
	provider, err := New(Config{APIKey: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return provider
}

func mustTool(t *testing.T, name, description string, schema any) litellm.Tool {
	t.Helper()
	tool, err := litellm.NewTool(name, description, schema)
	if err != nil {
		t.Fatalf("NewTool: %v", err)
	}
	return tool
}
