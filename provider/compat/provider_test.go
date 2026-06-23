package compat

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
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

func TestChatBuildsRequestAndConvertsResponse(t *testing.T) {
	var capturedBody map[string]any
	provider, err := New(Config{
		APIKey:  "test-key",
		BaseURL: "https://compat.example/v1",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if got := req.Header.Get("Authorization"); got != "Bearer test-key" {
				t.Fatalf("Authorization = %q", got)
			}
			if req.URL.String() != "https://compat.example/v1/chat/completions" {
				t.Fatalf("url = %s", req.URL.String())
			}
			if err := json.NewDecoder(req.Body).Decode(&capturedBody); err != nil {
				t.Fatalf("decode request body: %v", err)
			}
			return jsonResponse(http.StatusOK, testgolden.ReadFixtureString(t, "../../testdata/compat/chat_response.json")), nil
		}),
	}, Spec{
		Name: "testcompat",
		Auth: AuthSpec{APIKeyRequired: true},
		Request: RequestSpec{
			SupportsJSONSchema: true,
			AllowedProviderOptions: map[string]struct{}{
				"extra_body": {},
			},
		},
		Response: ResponseSpec{
			ModelFromResponse:         true,
			HasCompletionTokenDetails: true,
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	tool := mustTool(t, "lookup", "Lookup.", map[string]any{"type": "object"})
	resp, err := provider.Chat(context.Background(), &litellm.Request{
		Model: "m",
		Messages: []litellm.Message{
			litellm.User(litellm.Text("hi"), litellm.ImageURL("https://example.test/image.png")),
			litellm.Assistant(litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: litellm.MustJSONRaw(map[string]any{"q": "x"})}),
			litellm.ToolResultText("call_1", "ok"),
		},
		Tools:           []litellm.Tool{tool},
		ProviderOptions: litellm.ProviderOptions{"extra_body": true},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if capturedBody["model"] != "m" || capturedBody["extra_body"] != true {
		t.Fatalf("captured body = %#v", capturedBody)
	}
	if _, ok := capturedBody["messages"].([]any); !ok {
		t.Fatalf("messages not encoded as array: %#v", capturedBody["messages"])
	}
	if resp.Model != "provider-model" || resp.Text() != "hello" || resp.Reasoning() != "think" {
		t.Fatalf("response model/text/reasoning = %q/%q/%q", resp.Model, resp.Text(), resp.Reasoning())
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Arguments) != `{"q":"x"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if resp.Usage.ReasoningTokens != 2 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestChatRetriesWhenRetryPolicyIsConfigured(t *testing.T) {
	var attempts int
	provider, err := New(Config{
		BaseURL: "https://compat.example/v1",
		Retry:   &retry.Policy{MaxAttempts: 2, InitialDelay: 1},
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			attempts++
			if attempts == 1 {
				return jsonResponse(http.StatusTooManyRequests, `{"error":"retry"}`), nil
			}
			return jsonResponse(http.StatusOK, `{
				"model":"m",
				"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]
			}`), nil
		}),
	}, Spec{Name: "retrycompat"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if attempts != 2 || resp.Text() != "ok" {
		t.Fatalf("attempts/text = %d/%q", attempts, resp.Text())
	}
}

func TestNewRejectsAmbiguousTransportConfig(t *testing.T) {
	_, err := New(Config{
		BaseURL:    "https://compat.example/v1",
		HTTPClient: roundTripFunc(nil),
		Transport:  roundTripperFunc(nil),
	}, Spec{Name: "testcompat"})
	if err == nil || !strings.Contains(err.Error(), "HTTPClient and Transport are mutually exclusive") {
		t.Fatalf("expected HTTPClient/Transport error, got %v", err)
	}

	_, err = New(Config{
		BaseURL:    "https://compat.example/v1",
		HTTPClient: roundTripFunc(nil),
		Retry:      retry.DefaultPolicy(),
	}, Spec{Name: "testcompat"})
	if err == nil || !strings.Contains(err.Error(), "Retry cannot be used with a custom HTTPClient") {
		t.Fatalf("expected HTTPClient/Retry error, got %v", err)
	}
}

func TestRejectsUnknownProviderOptionByDefault(t *testing.T) {
	provider, err := New(Config{BaseURL: "https://compat.example", HTTPClient: roundTripFunc(nil)}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:           "m",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: litellm.ProviderOptions{"unknown": true},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported provider option") {
		t.Fatalf("expected unknown option error, got %v", err)
	}
}

func TestChatReturnsStructuredValidationError(t *testing.T) {
	provider, err := New(Config{BaseURL: "https://compat.example", HTTPClient: roundTripFunc(nil)}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:           "m",
		Messages:        []litellm.Message{litellm.UserText("hi")},
		ProviderOptions: litellm.ProviderOptions{"unknown": true},
	})
	if err == nil || !litellm.IsValidationError(err) {
		t.Fatalf("expected structured validation error, got %v", err)
	}
}

func TestRejectsStopSequencesAboveProviderLimit(t *testing.T) {
	provider, err := New(Config{BaseURL: "https://compat.example", HTTPClient: roundTripFunc(nil)}, Spec{
		Name:    "glm",
		Request: RequestSpec{MaxStopSequences: 1},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Stop:     []string{"a", "b"},
	})
	if err == nil || !strings.Contains(err.Error(), "stop supports at most 1 sequence") {
		t.Fatalf("expected stop limit error, got %v", err)
	}
}

func TestReasoningBlockHistoryRequiresProviderMapping(t *testing.T) {
	provider, err := New(Config{BaseURL: "https://compat.example", HTTPClient: roundTripFunc(nil)}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.Assistant(litellm.ReasoningBlock{Text: "think"})},
	})
	if err == nil || !strings.Contains(err.Error(), "ReasoningBlock history is not supported") {
		t.Fatalf("expected reasoning history error, got %v", err)
	}
}

func TestReasoningBlockExtraRoundTripsThroughConfiguredField(t *testing.T) {
	var capturedBody map[string]any
	provider, err := New(Config{
		BaseURL: "https://compat.example",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if err := json.NewDecoder(req.Body).Decode(&capturedBody); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			return jsonResponse(http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`), nil
		}),
	}, Spec{
		Name: "minimax",
		Response: ResponseSpec{
			ReasoningFields: []string{"reasoning_details"},
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model: "m",
		Messages: []litellm.Message{
			litellm.Assistant(litellm.ReasoningBlock{
				Text:  "step 1",
				Extra: json.RawMessage(`[{"type":"reasoning.text","text":"step 1"}]`),
			}),
		},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	messages := capturedBody["messages"].([]any)
	details := messages[0].(map[string]any)["reasoning_details"].([]any)
	if len(details) != 1 || details[0].(map[string]any)["text"] != "step 1" {
		t.Fatalf("reasoning_details = %#v", details)
	}
}

func TestResponseReasoningBlocksRoundTripThroughConfiguredField(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{
				"choices":[{
					"message":{
						"content":"ok",
						"reasoning_details":[{"type":"reasoning.text","text":"step 1"}]
					}
				}]
			}`), nil
		}),
	}, Spec{
		Name: "minimax",
		Response: ResponseSpec{
			ReasoningFields: []string{"reasoning_details"},
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	var capturedBody map[string]any
	provider.cfg.HTTPClient = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(req.Body).Decode(&capturedBody); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		return jsonResponse(http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`), nil
	})
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.Assistant(resp.Blocks...)},
	})
	if err != nil {
		t.Fatalf("round-trip Chat: %v", err)
	}
	messages := capturedBody["messages"].([]any)
	details := messages[0].(map[string]any)["reasoning_details"].([]any)
	if len(details) != 1 || details[0].(map[string]any)["text"] != "step 1" {
		t.Fatalf("reasoning_details = %#v", details)
	}
}

func TestChatRejectsUnsupportedResponseContent(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{
				"choices":[{
					"message":{"content":[{"type":"audio","url":"https://example.test/a.wav"}]},
					"finish_reason":"stop"
				}]
			}`), nil
		}),
	}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported content part type") {
		t.Fatalf("expected unsupported content error, got %v", err)
	}
}

func TestChatRejectsInvalidToolCallArguments(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{
				"choices":[{
					"message":{"tool_calls":[{
						"id":"call_1",
						"type":"function",
						"function":{"name":"lookup","arguments":"{\"q\":"}
					}]},
					"finish_reason":"tool_calls"
				}]
			}`), nil
		}),
	}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "arguments are not valid JSON") {
		t.Fatalf("expected invalid arguments error, got %v", err)
	}
}

func TestChatDecodeErrorIsStructuredProviderError(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return jsonResponse(http.StatusOK, `{"choices":[`), nil
		}),
	}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "decode response") || !litellm.IsProviderError(err) {
		t.Fatalf("expected structured decode provider error, got %v", err)
	}
}

func TestStreamCumulativeReasoningAndToolDeltas(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example/v1",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(testgolden.ReadFixtureString(t, "../../testdata/compat/minimax_stream.sse")), nil
		}),
	}, Spec{
		Name: "minimax",
		Stream: StreamSpec{
			ReasoningFields:     []string{"reasoning_content"},
			ReasoningCumulative: true,
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "minimax-text-01",
		Messages: []litellm.Message{litellm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Reasoning() != "ab" || resp.Text() != "hi" {
		t.Fatalf("reasoning/text = %q/%q", resp.Reasoning(), resp.Text())
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Arguments) != `{"q":"x"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if resp.Usage.InputTokens != 1 || resp.Usage.OutputTokens != 2 || resp.FinishReason != litellm.FinishReasonToolCall {
		t.Fatalf("usage/finish = %+v/%q", resp.Usage, resp.FinishReason)
	}
}

func TestStreamPrependsStrictToolOmittedWarning(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example/v1",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse("data: [DONE]\n\n"), nil
		}),
	}, Spec{
		Name:     "strictless",
		Features: FeatureSpec{StrictTools: StrictToolsOmit},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	tool := mustTool(t, "lookup", "Lookup.", map[string]any{"type": "object"})
	tool.Strict = litellm.StrictEnabled
	stream, err := provider.Stream(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Tools:    []litellm.Tool{tool},
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	defer stream.Close()
	event, err := stream.Next()
	if err != nil {
		t.Fatalf("Next returned error: %v", err)
	}
	warning, ok := event.(litellm.WarningEvent)
	if !ok {
		t.Fatalf("first event = %#v, want WarningEvent", event)
	}
	if warning.Warning.Code != "request.strict_tool_omitted" || warning.Warning.Provider != "strictless" {
		t.Fatalf("warning = %#v", warning.Warning)
	}
}

func TestThinkingMapperMustEmitFields(t *testing.T) {
	provider, err := New(Config{BaseURL: "https://compat.example", HTTPClient: roundTripFunc(nil)}, Spec{
		Name: "empty-thinking",
		Request: RequestSpec{
			Thinking: func(*litellm.Thinking, string) (map[string]any, error) {
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{
		Model:    "m",
		Messages: []litellm.Message{litellm.UserText("hi")},
		Thinking: &litellm.Thinking{Mode: litellm.ThinkingEnabled},
	})
	if err == nil || !strings.Contains(err.Error(), "thinking mapper produced no fields") {
		t.Fatalf("expected empty thinking mapper error, got %v", err)
	}
}

func TestStreamCumulativeReasoningRejectsRewrite(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example/v1",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(strings.Join([]string{
				`data: {"choices":[{"index":0,"delta":{"reasoning_content":"abc"}}]}`,
				`data: {"choices":[{"index":0,"delta":{"reasoning_content":"ax"}}]}`,
				``,
			}, "\n")), nil
		}),
	}, Spec{Name: "minimax", Stream: StreamSpec{ReasoningFields: []string{"reasoning_content"}, ReasoningCumulative: true}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	_, err = litellm.Collect(stream)
	if err == nil || !strings.Contains(err.Error(), "cumulative reasoning stream changed") {
		t.Fatalf("expected cumulative reasoning error, got %v", err)
	}
}

func TestStreamRejectsMalformedToolCall(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "non_object",
			body: `data: {"choices":[{"index":0,"delta":{"tool_calls":["bad"]}}]}`,
			want: "tool_call must be an object",
		},
		{
			name: "non_string_arguments",
			body: `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":123}}]}}]}`,
			want: "arguments must be string",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := New(Config{
				BaseURL: "https://compat.example/v1",
				HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					return streamResponse(tt.body), nil
				}),
			}, Spec{Name: "strict"})
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}
			stream, err := provider.Stream(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
			if err != nil {
				t.Fatalf("Stream returned error: %v", err)
			}
			_, err = stream.Next()
			if err == nil || !strings.Contains(err.Error(), tt.want) || !litellm.IsProviderError(err) {
				t.Fatalf("expected malformed tool call provider error containing %q, got %v", tt.want, err)
			}
		})
	}
}

func TestStreamRejectsEOFBeforeDoneSentinel(t *testing.T) {
	provider, err := New(Config{
		BaseURL: "https://compat.example/v1",
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return streamResponse(`data: {"choices":[{"delta":{"content":"partial"}}]}`), nil
		}),
	}, Spec{Name: "strict"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := provider.Stream(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	_, err = litellm.Collect(stream)
	if err == nil || !strings.Contains(err.Error(), "before [DONE]") || !litellm.IsProviderError(err) {
		t.Fatalf("expected truncated stream error, got %v", err)
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

func mustTool(t *testing.T, name, description string, schema any) litellm.Tool {
	t.Helper()
	tool, err := litellm.NewTool(name, description, schema)
	if err != nil {
		t.Fatalf("NewTool: %v", err)
	}
	return tool
}
