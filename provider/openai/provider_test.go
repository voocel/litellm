package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

type httpFunc func(*http.Request) (*http.Response, error)

func (f httpFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

// captured records the last request a test provider sent.
type captured struct {
	req  *http.Request
	body []byte
}

// testProvider answers every request with reply; API key "key" is the default.
func testProvider(t *testing.T, cfg Config, reply string) (*Provider, *captured) {
	t.Helper()
	got := &captured{}
	if cfg.APIKey == "" {
		cfg.APIKey = "key"
	}
	cfg.HTTPClient = httpFunc(func(req *http.Request) (*http.Response, error) {
		got.req = req
		got.body, _ = io.ReadAll(req.Body)
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(reply))}, nil
	})
	p, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return p, got
}

func streamResponse(body string) *http.Response {
	return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(body))}
}

func assertJSON(t *testing.T, got []byte, want string) {
	t.Helper()
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("decode got: %v\n%s", err, got)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("decode want: %v", err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("JSON mismatch\ngot:  %s\nwant: %s", got, want)
	}
}

func providerOptions(t *testing.T, values map[string]any) litellm.ProviderOptions {
	t.Helper()
	o, err := litellm.NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

const chatReply = `{"model":"gpt-4.1","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`

func TestNew(t *testing.T) {
	if _, err := New(Config{APIKey: "key", API: "legacy"}); err == nil || !strings.Contains(err.Error(), `api must be "chat" or "responses"`) {
		t.Fatalf("unknown API: err = %v", err)
	}
	if _, err := New(Config{}); litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation {
		t.Fatalf("missing key: err = %v", err)
	}
}

func TestEndpointsAndHeaders(t *testing.T) {
	req := &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}}
	tests := []struct {
		api, baseURL, reply, url string
	}{
		{"", "", chatReply, "https://api.openai.com/v1/chat/completions"},
		{APIResponses, "", `{"status":"completed","output":[]}`, "https://api.openai.com/v1/responses"},
		{APIChat, "https://proxy.test/openai/v1/", chatReply, "https://proxy.test/openai/v1/chat/completions"},
	}
	for _, test := range tests {
		p, got := testProvider(t, Config{API: test.api, BaseURL: test.baseURL, UserAgent: "agent/1", Headers: map[string]string{"X-Tenant": "acme"}}, test.reply)
		if _, err := p.Chat(context.Background(), req); err != nil {
			t.Fatalf("%s: %v", test.url, err)
		}
		h := got.req.Header
		if got.req.URL.String() != test.url || h.Get("Authorization") != "Bearer key" || h.Get("User-Agent") != "agent/1" || h.Get("X-Tenant") != "acme" {
			t.Fatalf("request = %s %v", got.req.URL, h)
		}
	}
}

func TestCapabilities(t *testing.T) {
	for api, want := range map[string][]string{APIChat: chatOptions, APIResponses: responsesOptions} {
		p, _ := testProvider(t, Config{API: api}, "")
		caps := p.Capabilities()
		if caps.MaxTokensRequired || !caps.ThinkingEffort || !caps.DisableThinking || !slices.IsSorted(caps.ProviderOptions) || !reflect.DeepEqual(caps.ProviderOptions, want) {
			t.Fatalf("%s: capabilities = %+v", api, caps)
		}
		caps.ProviderOptions[0] = "mutated"
		if p.Capabilities().ProviderOptions[0] == "mutated" {
			t.Fatalf("%s: capabilities share the option list", api)
		}
	}
}

// Options of the other API point to the Config.API that accepts them; unknown
// options are rejected by the selected API.
func TestProviderOptionsFollowAPI(t *testing.T) {
	tests := []struct {
		api, option, wantErr string
	}{
		{APIChat, ProviderOptionTruncation, `provider option "truncation" requires Config.API "responses"`},
		{APIResponses, ProviderOptionLogitBias, `provider option "logit_bias" requires Config.API "chat"`},
		{APIChat, "unknown", `unsupported provider option "unknown"`},
		{APIResponses, "unknown", `unsupported provider option "unknown"`},
		{APIChat, ProviderOptionStore, ""},
		{APIResponses, ProviderOptionStore, ""},
	}
	for _, test := range tests {
		reply := chatReply
		if test.api == APIResponses {
			reply = `{"status":"completed","output":[]}`
		}
		p, _ := testProvider(t, Config{API: test.api}, reply)
		_, err := p.Chat(context.Background(), &litellm.Request{
			Model:           "m",
			Messages:        []litellm.Message{litellm.UserText("hi")},
			ProviderOptions: providerOptions(t, map[string]any{test.option: true}),
		})
		if test.wantErr == "" && err != nil || test.wantErr != "" && (litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || !strings.Contains(err.Error(), test.wantErr)) {
			t.Errorf("%s %s: err = %v, want %q", test.api, test.option, err, test.wantErr)
		}
	}
}

func TestChatRequest(t *testing.T) {
	p, got := testProvider(t, Config{}, chatReply)
	tool, err := litellm.NewTool("lookup", "Lookup data.", map[string]any{
		"type":       "object",
		"properties": map[string]any{"q": map[string]any{"type": "string"}},
		"required":   []string{"q"},
	})
	if err != nil {
		t.Fatal(err)
	}
	tool.Strict = new(true)
	_, err = p.Chat(context.Background(), &litellm.Request{
		Model:       "gpt-4.1",
		MaxTokens:   new(256),
		Temperature: new(0.2),
		Messages: []litellm.Message{
			litellm.System("You are helpful."),
			litellm.User(litellm.Text("describe"), litellm.ImageURL("https://example.test/image.png")),
			litellm.Assistant(litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: `{"q":"x"}`}),
			litellm.ToolResultText("call_1", "result"),
		},
		Tools: []litellm.Tool{tool},
		ProviderOptions: providerOptions(t, map[string]any{
			ProviderOptionFrequencyPenalty: 0.4,
			ProviderOptionMetadata:         map[string]any{"tenant": "acme"},
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	testgolden.AssertJSONBytes(t, "../../testdata/openai/chat_request_basic.golden.json", got.body)
}

// Chat marks cache breakpoints on content parts; OpenAI sets the TTL for the
// whole request, through prompt_cache_options.
func TestChatPromptCacheBreakpoint(t *testing.T) {
	p, got := testProvider(t, Config{}, chatReply)
	cached := &litellm.CacheControl{}
	_, err := p.Chat(context.Background(), &litellm.Request{
		Model: "m",
		Messages: []litellm.Message{
			litellm.User(litellm.TextBlock{Text: "context", Cache: cached}),
			litellm.Assistant(litellm.ToolUseBlock{ID: "call_1", Name: "f", Arguments: `{}`}),
			{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: "call_1", Content: []litellm.Block{litellm.Text("r")}, Cache: cached}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, got.body, `{"model":"m","messages":[
		{"role":"user","content":[{"type":"text","text":"context","prompt_cache_breakpoint":{"mode":"explicit"}}]},
		{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"r","prompt_cache_breakpoint":{"mode":"explicit"}}]}
	]}`)
}

func TestChatStream(t *testing.T) {
	p, got := testProvider(t, Config{}, testgolden.ReadFixtureString(t, "../../testdata/openai/chat_stream.sse"))
	stream, err := p.Stream(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, got.body, `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	if got.req.Header.Get("Accept") != "text/event-stream" {
		t.Fatalf("Accept = %q", got.req.Header.Get("Accept"))
	}
	want := &litellm.Response{
		Blocks: []litellm.Block{
			litellm.TextBlock{Text: "hello"},
			litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: `{"q":"x"}`},
		},
		Usage:           litellm.Usage{InputTokens: 4, OutputTokens: 3},
		Model:           "gpt-4.1",
		Provider:        "openai",
		FinishReason:    litellm.FinishReasonToolCall,
		FinishReasonRaw: "tool_calls",
	}
	if !reflect.DeepEqual(resp, want) {
		t.Fatalf("response = %#v\nwant %#v", resp, want)
	}
}

// Models are listed from /models whichever API is selected.
func TestListModels(t *testing.T) {
	p, err := New(Config{API: APIResponses, APIKey: "key", HTTPClient: httpFunc(func(req *http.Request) (*http.Response, error) {
		if req.URL.String() != "https://api.openai.com/v1/models" || req.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("%s with %q", req.URL, req.Header.Get("Authorization"))
		}
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"data":[{"id":"gpt-x","created":1750000000}]}`))}, nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.ListModels(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "gpt-x" {
		t.Fatalf("%+v, %v", models, err)
	}
}
