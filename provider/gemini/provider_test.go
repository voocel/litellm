package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestBuildRequestGolden(t *testing.T) {
	data, err := buildRequest(&litellm.Request{
		Model: "gemini-3-pro",
		Messages: []litellm.Message{
			litellm.System("be concise"),
			litellm.User(
				litellm.Text("weather?"),
				litellm.ImageBlock{FileURI: "gs://bucket/image.png", MIME: "image/png"},
			),
			litellm.Assistant(
				litellm.ReasoningBlock{Text: "Need weather.", Signature: "sig-think"},
				litellm.ToolUseBlock{ID: "call_weather", Name: "get_weather", Arguments: json.RawMessage(`{"city":"Paris"}`), Signature: "sig-call"},
			),
			litellm.ToolResultText("call_weather", `{"temp":"15C"}`),
		},
		Tools: []litellm.Tool{mustTool(t, "get_weather", "Get weather.", map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		})},
		Thinking: &litellm.Thinking{Effort: "low", IncludeOutput: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	testgolden.AssertJSONBytes(t, "../../testdata/gemini/request_multimodal_tools.golden.json", data)
}

func TestBuildRequestThinking(t *testing.T) {
	for _, test := range []struct {
		name     string
		thinking *litellm.Thinking
		want     string
	}{
		{"vendor default", nil, ``},
		{"enabled", &litellm.Thinking{}, `{"thinkingConfig":{}}`},
		{"disabled", &litellm.Thinking{Mode: litellm.ThinkingDisabled}, `{"thinkingConfig":{"thinkingBudget":0}}`},
		{"effort", &litellm.Thinking{Effort: "high"}, `{"thinkingConfig":{"thinkingLevel":"high"}}`},
		{"budget", &litellm.Thinking{BudgetTokens: new(1024)}, `{"thinkingConfig":{"thinkingBudget":1024}}`},
		{"include output", &litellm.Thinking{Effort: "low", IncludeOutput: true}, `{"thinkingConfig":{"thinkingLevel":"low","includeThoughts":true}}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := build(t, litellm.Request{Messages: []litellm.Message{litellm.UserText("hi")}, Thinking: test.thinking})
			assertField(t, body, "generationConfig", test.want)
		})
	}
}

func TestBuildRequestMergesProviderOptions(t *testing.T) {
	body := build(t, litellm.Request{
		Messages:    []litellm.Message{litellm.UserText("hi")},
		Temperature: new(0.5),
		Tools:       []litellm.Tool{mustTool(t, "lookup", "", nil)},
		ToolChoice:  &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto},
		ProviderOptions: mustOptions(t, map[string]any{
			ProviderOptionGenerationConfig: map[string]any{"topK": 40},
			ProviderOptionTools:            []any{map[string]any{"googleSearch": map[string]any{}}},
			ProviderOptionToolConfig:       map[string]any{"retrievalConfig": map[string]any{"languageCode": "en"}},
			ProviderOptionSafetySettings:   []any{map[string]any{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}},
			ProviderOptionCachedContent:    "cachedContents/abc",
		}),
	})
	assertField(t, body, "generationConfig", `{"temperature":0.5,"topK":40}`)
	assertField(t, body, "tools", `[{"functionDeclarations":[{"name":"lookup"}]},{"googleSearch":{}}]`)
	assertField(t, body, "toolConfig", `{"functionCallingConfig":{"mode":"AUTO"},"retrievalConfig":{"languageCode":"en"}}`)
	assertField(t, body, "safetySettings", `[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}]`)
	assertField(t, body, "cachedContent", `"cachedContents/abc"`)
}

func TestBuildRequestToolChoice(t *testing.T) {
	strict := mustTool(t, "lookup", "", nil)
	strict.Strict = litellm.StrictEnabled
	for _, test := range []struct {
		name   string
		strict bool
		choice *litellm.ToolChoice
		want   string
	}{
		{"default", false, nil, ``},
		{"auto", false, &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto}, `{"functionCallingConfig":{"mode":"AUTO"}}`},
		{"none", false, &litellm.ToolChoice{Mode: litellm.ToolChoiceNone}, `{"functionCallingConfig":{"mode":"NONE"}}`},
		{"required", false, &litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}, `{"functionCallingConfig":{"mode":"ANY"}}`},
		{"named", false, &litellm.ToolChoice{Name: "lookup"}, `{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["lookup"]}}`},
		{"strict default", true, nil, `{"functionCallingConfig":{"mode":"VALIDATED"}}`},
		{"strict auto", true, &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto}, `{"functionCallingConfig":{"mode":"VALIDATED"}}`},
		{"strict required", true, &litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}, `{"functionCallingConfig":{"mode":"ANY"}}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			tool := mustTool(t, "lookup", "", nil)
			if test.strict {
				tool = strict
			}
			body := build(t, litellm.Request{Messages: []litellm.Message{litellm.UserText("hi")}, Tools: []litellm.Tool{tool}, ToolChoice: test.choice})
			assertField(t, body, "toolConfig", test.want)
		})
	}
}

func TestBuildRequestPreservesRawJSON(t *testing.T) {
	schema := `{"type":"object","properties":{"note":{"type":["string","null"]}},"additionalProperties":false}`
	format, err := litellm.NewResponseFormatJSONSchema("out", "", json.RawMessage(schema), litellm.StrictDefault)
	if err != nil {
		t.Fatal(err)
	}
	data, err := buildRequest(&litellm.Request{
		Messages: []litellm.Message{
			litellm.Assistant(litellm.ToolUseBlock{ID: "c1", Name: "lookup", Arguments: json.RawMessage(`{"n":12345678901234567890}`)}),
			litellm.ToolResultText("c1", `{"ok":true}`),
		},
		Tools:          []litellm.Tool{mustTool(t, "lookup", "", json.RawMessage(schema))},
		ResponseFormat: format,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`"args":{"n":12345678901234567890}`,
		`"response":{"ok":true}`,
		`"parametersJsonSchema":` + schema,
		`"responseMimeType":"application/json","responseJsonSchema":` + schema,
	} {
		if !strings.Contains(string(data), want) {
			t.Errorf("body missing %s:\n%s", want, data)
		}
	}
}

func TestBuildRequestToolResultsAndTurns(t *testing.T) {
	body := build(t, litellm.Request{Messages: []litellm.Message{
		litellm.UserText("go"),
		litellm.Assistant(
			litellm.ToolUseBlock{ID: "a", Name: "first"},
			litellm.ToolUseBlock{ID: "b", Name: "second", Signature: SkipThoughtSignatureValidator},
		),
		litellm.ToolResultText("a", "plain"),
		{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: "b", Content: []litellm.Block{litellm.Text("boom")}, IsError: true}}},
		litellm.UserText("continue"),
	}})
	// Parallel responses and the following text share one user turn.
	assertField(t, body, "contents", `[
		{"role":"user","parts":[{"text":"go"}]},
		{"role":"model","parts":[
			{"functionCall":{"id":"a","name":"first","args":{}}},
			{"functionCall":{"id":"b","name":"second","args":{}},"thoughtSignature":"skip_thought_signature_validator"}
		]},
		{"role":"user","parts":[
			{"functionResponse":{"id":"a","name":"first","response":{"result":"plain"}}},
			{"functionResponse":{"id":"b","name":"second","response":{"error":"boom"}}},
			{"text":"continue"}
		]}
	]`)
}

func TestBuildRequestReplaysSignatures(t *testing.T) {
	body := build(t, litellm.Request{Messages: []litellm.Message{
		litellm.UserText("go"),
		litellm.Assistant(
			litellm.ReasoningBlock{Signature: "r"},
			litellm.TextBlock{Text: "answer", Signature: "t"},
			litellm.TextBlock{Signature: "e"},
		),
	}})
	// Text is the part's data, so signature-only parts keep an empty text.
	assertField(t, body, "contents", `[
		{"role":"user","parts":[{"text":"go"}]},
		{"role":"model","parts":[
			{"text":"","thought":true,"thoughtSignature":"r"},
			{"text":"answer","thoughtSignature":"t"},
			{"text":"","thoughtSignature":"e"}
		]}
	]`)
}

func TestBuildRequestErrors(t *testing.T) {
	strict := mustTool(t, "a", "", nil)
	strict.Strict = litellm.StrictEnabled
	loose := mustTool(t, "b", "", nil)
	loose.Strict = litellm.StrictDisabled
	for _, test := range []struct {
		name string
		req  litellm.Request
		want string
	}{
		{"unknown option", litellm.Request{ProviderOptions: mustOptions(t, map[string]any{"topK": 1})}, `unsupported provider option "topK"`},
		{"orphan tool result", litellm.Request{Messages: []litellm.Message{litellm.ToolResultText("missing", "x")}}, "no preceding tool use"},
		{"non-object args", litellm.Request{Messages: []litellm.Message{litellm.Assistant(litellm.ToolUseBlock{ID: "c", Name: "n", Arguments: json.RawMessage(`[1]`)})}}, "must be a JSON object"},
		{"inline image without MIME", litellm.Request{Messages: []litellm.Message{litellm.User(litellm.ImageBlock{Data: []byte{1}})}}, "requires MIME"},
		{"mixed strict tools", litellm.Request{Tools: []litellm.Tool{strict, loose}}, "cannot mix strict"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := buildRequest(&test.req); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err = %v, want %q", err, test.want)
			}
		})
	}
}

func TestChatConvertsResponse(t *testing.T) {
	provider := testProvider(t, func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != "/v1beta/models/gemini-3-pro:generateContent" || req.URL.RawQuery != "" {
			t.Errorf("url = %s", req.URL)
		}
		if got := req.Header.Get("x-goog-api-key"); got != "test-key" {
			t.Errorf("x-goog-api-key = %q", got)
		}
		return jsonResponse(http.StatusOK, `{
			"candidates":[{"content":{"parts":[
				{"text":"thin","thought":true},
				{"text":"king","thought":true,"thoughtSignature":"sig-think"},
				{"text":"ans"},
				{"text":"wer"},
				{"thoughtSignature":"sig-call","functionCall":{"id":"call_1","name":"lookup","args":{"q":"x"}}},
				{"functionCall":{"name":"noop"}}
			]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"thoughtsTokenCount":2,"totalTokenCount":9,"cachedContentTokenCount":1}
		}`), nil
	})
	resp, err := provider.Chat(context.Background(), &litellm.Request{Model: "gemini-3-pro", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	generated, ok := resp.Blocks[len(resp.Blocks)-1].(litellm.ToolUseBlock)
	if !ok || !strings.HasPrefix(generated.ID, "call_") {
		t.Fatalf("last block = %#v", resp.Blocks[len(resp.Blocks)-1])
	}
	want := []litellm.Block{
		litellm.ReasoningBlock{Text: "thinking", Signature: "sig-think"},
		litellm.TextBlock{Text: "answer"},
		litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`), Signature: "sig-call"},
		litellm.ToolUseBlock{ID: generated.ID, Name: "noop", Arguments: json.RawMessage(`{}`)},
	}
	if !reflect.DeepEqual(resp.Blocks, want) {
		t.Fatalf("blocks = %#v", resp.Blocks)
	}
	if resp.FinishReason != litellm.FinishReasonToolCall || resp.FinishReasonRaw != "STOP" {
		t.Fatalf("finish = %q/%q", resp.FinishReason, resp.FinishReasonRaw)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "gemini.tool_call_id_generated" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
	if *resp.Usage.InputTokens != 3 || *resp.Usage.OutputTokens != 6 || *resp.Usage.ReasoningTokens != 2 || *resp.Usage.CacheReadTokens != 1 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestChatReturnsBlockedResponses(t *testing.T) {
	for _, test := range []struct {
		name, body, raw string
	}{
		{"prompt", `{"promptFeedback":{"blockReason":"SAFETY"}}`, "SAFETY"},
		{"candidate", `{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}]}`, "PROHIBITED_CONTENT"},
	} {
		t.Run(test.name, func(t *testing.T) {
			provider := testProvider(t, func(*http.Request) (*http.Response, error) {
				return jsonResponse(http.StatusOK, test.body), nil
			})
			resp, err := provider.Chat(context.Background(), &litellm.Request{Model: "m"})
			if err != nil {
				t.Fatal(err)
			}
			if resp.FinishReason != litellm.FinishReasonSafety || resp.FinishReasonRaw != test.raw || len(resp.Blocks) != 0 {
				t.Fatalf("response = %+v", resp)
			}
		})
	}
}

func TestStreamEvents(t *testing.T) {
	provider := testProvider(t, func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != "/v1beta/models/gemini-3-pro:streamGenerateContent" || req.URL.RawQuery != "alt=sse" {
			t.Errorf("url = %s", req.URL)
		}
		if req.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("Accept = %q", req.Header.Get("Accept"))
		}
		return streamResponse(testgolden.ReadFixtureString(t, "../../testdata/gemini/stream.jsonl")), nil
	})
	stream, err := provider.Stream(context.Background(), &litellm.Request{Model: "gemini-3-pro"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	usage := litellm.Usage{InputTokens: new(3), OutputTokens: new(6), ReasoningTokens: new(2), TotalTokens: new(9), CacheReadTokens: new(1)}
	assertEvents(t, stream, []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.ReasoningBlock{}},
		litellm.ReasoningDelta{Index: 0, Text: "think"},
		litellm.BlockEnd{Index: 0, Block: litellm.ReasoningBlock{Signature: "sig-think"}},
		litellm.BlockStart{Index: 1, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 1, Text: "ans"},
		litellm.TextDelta{Index: 1, Text: "wer"},
		litellm.BlockEnd{Index: 1},
		litellm.BlockStart{Index: 2, Block: litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Signature: "sig-call"}},
		litellm.ToolUseDelta{Index: 2, Arguments: `{"q":"x"}`},
		litellm.BlockEnd{Index: 2},
		litellm.UsageEvent{Usage: usage},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "STOP", Provider: "gemini", Model: "gemini-3-pro"},
	})
}

func TestStreamEndings(t *testing.T) {
	for _, test := range []struct {
		name    string
		body    string
		finish  litellm.FinishReason
		wantErr func(error) bool
	}{
		{"prompt blocked", `data: {"promptFeedback":{"blockReason":"SAFETY"}}`, litellm.FinishReasonSafety, nil},
		{"eof before finish", `data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`, "", litellm.IsProviderError},
		{"mid-stream error", `data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}` + "\n" +
			`data: {"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}`, "", func(err error) bool {
			var e *litellm.Error
			return errors.As(err, &e) && e.StatusCode == http.StatusServiceUnavailable && e.Temporary && strings.Contains(e.Message, "overloaded")
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			resp, err := litellm.Collect(newStream(streamResponse(test.body), "m"))
			if test.wantErr != nil {
				if !test.wantErr(err) {
					t.Fatalf("err = %v", err)
				}
				return
			}
			if err != nil || resp.FinishReason != test.finish {
				t.Fatalf("resp = %+v, err = %v", resp, err)
			}
		})
	}
}

func TestStreamWarnsForGeneratedToolCallID(t *testing.T) {
	resp, err := litellm.Collect(newStream(streamResponse(`data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"noop"}}]},"finishReason":"STOP"}]}`), "m"))
	if err != nil {
		t.Fatal(err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || !strings.HasPrefix(calls[0].ID, "call_") || string(calls[0].Arguments) != `{}` {
		t.Fatalf("calls = %+v", calls)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "gemini.tool_call_id_generated" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

func TestUsageIncludesReasoningAndReadsOmittedCountsAsZero(t *testing.T) {
	var meta usageMetadata
	if err := json.Unmarshal([]byte(`{"promptTokenCount":10,"candidatesTokenCount":2,"thoughtsTokenCount":3,"totalTokenCount":15,"cachedContentTokenCount":4}`), &meta); err != nil {
		t.Fatal(err)
	}
	usage := convertUsage(&meta)
	if *usage.InputTokens != 10 || *usage.OutputTokens != 5 || *usage.ReasoningTokens != 3 || *usage.TotalTokens != 15 || *usage.CacheReadTokens != 4 || usage.CacheWriteTokens != nil {
		t.Fatalf("usage = %+v", usage)
	}
	// The API omits zero counts, e.g. no output after thinking hit MAX_TOKENS.
	meta = usageMetadata{}
	if err := json.Unmarshal([]byte(`{"promptTokenCount":10,"thoughtsTokenCount":3,"totalTokenCount":13}`), &meta); err != nil {
		t.Fatal(err)
	}
	usage = convertUsage(&meta)
	if *usage.OutputTokens != 3 || *usage.ReasoningTokens != 3 || *usage.CacheReadTokens != 0 {
		t.Fatalf("usage with omitted counts = %+v", usage)
	}
	if convertUsage(&usageMetadata{}).HasTokens() {
		t.Fatal("empty metadata became known")
	}
}

func TestCapabilities(t *testing.T) {
	caps := testProvider(t, nil).Capabilities()
	if !caps.Thinking || !caps.DisableThinking || !caps.ThinkingEffort || !caps.ThinkingBudget || !slices.IsSorted(caps.ProviderOptions) || !slices.Equal(caps.ProviderOptions, sortedOptions()) || len(caps.ProviderOptions) != len(providerOptions) {
		t.Fatalf("capabilities = %+v", caps)
	}
}

func TestRequestHeadersAndSecrets(t *testing.T) {
	const secret = "gemini-secret-key"
	provider, err := New(Config{
		APIKey:    secret,
		BaseURL:   "https://example.test",
		UserAgent: "app/1.0",
		Headers:   map[string]string{"X-Gateway": "edge"},
		HTTPClient: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.Header.Get("User-Agent") != "app/1.0" || req.Header.Get("X-Gateway") != "edge" {
				t.Errorf("headers = %v", req.Header)
			}
			return nil, fmt.Errorf("request to %s failed", req.URL)
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = provider.Chat(context.Background(), &litellm.Request{Model: "m"})
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatalf("err = %v", err)
	}
}

func TestListModels(t *testing.T) {
	provider := testProvider(t, func(req *http.Request) (*http.Response, error) {
		if req.URL.Path != "/v1beta/models" || req.URL.RawQuery != "" || req.Header.Get("x-goog-api-key") != "test-key" {
			t.Errorf("request = %s %v", req.URL, req.Header)
		}
		return jsonResponse(http.StatusOK, `{"models":[{"name":"models/gemini-3-pro","inputTokenLimit":10}]}`), nil
	})
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.ModelInfo{{ID: "gemini-3-pro", Name: "gemini-3-pro", Provider: "gemini", InputTokenLimit: 10}}
	if !reflect.DeepEqual(models, want) {
		t.Fatalf("models = %+v", models)
	}
}

// assertEvents reads stream to its end and compares the events.
func assertEvents(t *testing.T, stream litellm.Stream, want []litellm.Event) {
	t.Helper()
	var got []litellm.Event
	for {
		event, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		got = append(got, event)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("events:\n got %#v\nwant %#v", got, want)
	}
}

func build(t *testing.T, req litellm.Request) map[string]any {
	t.Helper()
	data, err := buildRequest(&req)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatal(err)
	}
	return body
}

// assertField compares body[key] with want JSON; empty want means absent.
func assertField(t *testing.T, body map[string]any, key, want string) {
	t.Helper()
	got, ok := body[key]
	if want == "" {
		if ok {
			t.Fatalf("%s = %v, want absent", key, got)
		}
		return
	}
	var expected any
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, expected) {
		data, _ := json.Marshal(got)
		t.Fatalf("%s = %s, want %s", key, data, want)
	}
}

func testProvider(t *testing.T, fn roundTripFunc) *Provider {
	t.Helper()
	provider, err := New(Config{APIKey: "test-key", BaseURL: "https://example.test", HTTPClient: fn})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
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
		t.Fatal(err)
	}
	return tool
}

func mustOptions(t *testing.T, values map[string]any) litellm.ProviderOptions {
	t.Helper()
	o, err := litellm.NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}
