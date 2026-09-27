package openaicompat_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
)

var plain = compattest.Spec(openaicompat.Spec{Name: "test", ReasoningFields: []string{"reasoning_content"}})

func allowUnknown(spec openaicompat.Spec) compattest.NewFunc {
	return func(cfg openaicompat.Config) (*openaicompat.Provider, error) {
		cfg.AllowUnknownProviderOptions = true
		return openaicompat.New(cfg, spec)
	}
}

func TestRequestBody(t *testing.T) {
	body := compattest.Body(t, plain, &litellm.Request{
		Model: "m",
		Messages: []litellm.Message{
			litellm.System("be brief"),
			litellm.User(litellm.Text("look"), litellm.ImageBlock{Data: []byte("png"), MIME: "image/png", Detail: "low"}),
			litellm.Assistant(litellm.ReasoningBlock{Text: "plan"}, litellm.Text("calling"), litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)}),
			litellm.ToolResult("call_1", litellm.Text("a"), litellm.Text("b")),
		},
		MaxTokens:   new(64),
		Temperature: new(0.5),
		TopP:        new(0.9),
		Stop:        []string{"END"},
		Tools: []litellm.Tool{
			{Name: "lookup", Description: "Lookup.", Parameters: litellm.Schema(`{"type":"object","properties":{"q":{"type":"string"}}}`), Strict: litellm.StrictEnabled},
			{Name: "ping", Strict: litellm.StrictDisabled},
		},
		ToolChoice:     &litellm.ToolChoice{Name: "lookup"},
		ResponseFormat: &litellm.ResponseFormat{Type: litellm.ResponseFormatJSONSchema, JSONSchema: &litellm.JSONSchema{Name: "out", Schema: litellm.Schema(`{"type":"object"}`)}},
	}, false)
	compattest.AssertJSON(t, body, `{
		"model": "m",
		"messages": [
			{"role": "system", "content": "be brief"},
			{"role": "user", "content": [
				{"type": "text", "text": "look"},
				{"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n", "detail": "low"}}
			]},
			{"role": "assistant", "reasoning_content": "plan", "content": "calling", "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "a\nb"}
		],
		"max_tokens": 64,
		"temperature": 0.5,
		"top_p": 0.9,
		"stop": ["END"],
		"tools": [
			{"type": "function", "function": {"name": "lookup", "description": "Lookup.", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}, "strict": true}},
			{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}, "strict": false}}
		],
		"tool_choice": {"type": "function", "function": {"name": "lookup"}},
		"response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}}}
	}`)
}

func TestRequestFields(t *testing.T) {
	thinking := openaicompat.Spec{Name: "test", Thinking: openaicompat.ThinkingType("enabled", false), Options: []string{"thinking", "tools"}}
	tests := []struct {
		name   string
		newFn  compattest.NewFunc
		req    litellm.Request
		stream bool
		want   string
	}{
		{name: "max tokens", newFn: plain, req: litellm.Request{MaxTokens: new(8)}, want: `{"max_tokens": 8, "max_completion_tokens": null}`},
		{name: "max tokens field", newFn: compattest.Spec(openaicompat.Spec{Name: "test", MaxTokensField: "max_completion_tokens"}), req: litellm.Request{MaxTokens: new(8)}, want: `{"max_tokens": null, "max_completion_tokens": 8}`},
		{name: "tool choice mode", newFn: plain, req: litellm.Request{ToolChoice: &litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}}, want: `{"tool_choice": "required"}`},
		{name: "json object", newFn: plain, req: litellm.Request{ResponseFormat: litellm.NewResponseFormatJSONObject()}, want: `{"response_format": {"type": "json_object"}}`},
		{name: "text format", newFn: plain, req: litellm.Request{ResponseFormat: litellm.NewResponseFormatText()}, want: `{"response_format": null}`},
		{name: "stream", newFn: plain, stream: true, want: `{"stream": true, "stream_options": {"include_usage": true}}`},
		{name: "omit stream options", newFn: compattest.Spec(openaicompat.Spec{Name: "test", OmitStreamOptions: true}), stream: true, want: `{"stream": true, "stream_options": null}`},
		{name: "fields", newFn: compattest.Spec(openaicompat.Spec{Name: "test", Fields: map[string]any{"reasoning_split": true}}), stream: true, want: `{"reasoning_split": true}`},
		{name: "allowed option", newFn: compattest.Spec(openaicompat.Spec{Name: "test", Options: []string{"seed"}}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"seed": json.RawMessage(`7`)}}, want: `{"seed": 7}`},
		{name: "unknown option allowed", newFn: allowUnknown(openaicompat.Spec{Name: "test"}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"min_p": json.RawMessage(`0.05`)}}, want: `{"min_p": 0.05}`},
		{name: "single output", newFn: allowUnknown(openaicompat.Spec{Name: "test"}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"n": json.RawMessage(`1`)}}, want: `{"n": 1}`},
		{
			name:  "option merges into generated object",
			newFn: compattest.Spec(thinking),
			req:   litellm.Request{Thinking: &litellm.Thinking{}, ProviderOptions: map[string]json.RawMessage{"thinking": json.RawMessage(`{"clear_thinking":false}`)}},
			want:  `{"thinking": {"type": "enabled", "clear_thinking": false}}`,
		},
		{
			name:  "option appends to generated array",
			newFn: compattest.Spec(thinking),
			req:   litellm.Request{Tools: []litellm.Tool{{Name: "ping"}}, ProviderOptions: map[string]json.RawMessage{"tools": json.RawMessage(`[{"type":"web_search"}]`)}},
			want:  `{"tools": [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}, {"type": "web_search"}]}`,
		},
		{
			name:  "option appends messages",
			newFn: allowUnknown(openaicompat.Spec{Name: "test"}),
			req:   litellm.Request{ProviderOptions: map[string]json.RawMessage{"messages": json.RawMessage(`[{"role":"user","content":"more"}]`)}},
			want:  `{"messages": [{"role": "user", "content": "hi"}, {"role": "user", "content": "more"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := tt.req
			req.Model, req.Messages = "m", []litellm.Message{litellm.UserText("hi")}
			compattest.AssertFields(t, compattest.Body(t, tt.newFn, &req, tt.stream), tt.want)
		})
	}
}

func TestRequestErrors(t *testing.T) {
	tests := []struct {
		name  string
		newFn compattest.NewFunc
		req   litellm.Request
		want  string
	}{
		{name: "unknown option", newFn: compattest.Spec(openaicompat.Spec{Name: "test", Options: []string{"seed"}}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"top_k": json.RawMessage(`1`)}}, want: `unsupported provider option "top_k"`},
		{name: "option replaces generated scalar", newFn: allowUnknown(openaicompat.Spec{Name: "test"}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"model": json.RawMessage(`"x"`)}}, want: `provider option "model" conflicts`},
		{name: "several outputs", newFn: allowUnknown(openaicompat.Spec{Name: "test"}), req: litellm.Request{ProviderOptions: map[string]json.RawMessage{"n": json.RawMessage(`2`)}}, want: `provider option "n" must be 1`},
		{name: "cache hook", newFn: compattest.Spec(openaicompat.Spec{Name: "test", Cache: func(*litellm.CacheControl) (map[string]any, error) { return nil, errors.New("ttl is not supported") }}), req: litellm.Request{Messages: []litellm.Message{litellm.User(litellm.TextBlock{Text: "hi", Cache: &litellm.CacheControl{TTL: litellm.CacheTTL1h}})}}, want: "ttl is not supported"},
		{name: "inline image without MIME", newFn: plain, req: litellm.Request{Messages: []litellm.Message{litellm.User(litellm.ImageBlock{Data: []byte("x")})}}, want: "inline image requires MIME"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := tt.req
			req.Model = "m"
			if req.Messages == nil {
				req.Messages = []litellm.Message{litellm.UserText("hi")}
			}
			err := compattest.Err(t, tt.newFn, &req)
			if !litellm.IsValidationError(err) || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want validation error containing %q", err, tt.want)
			}
		})
	}
}

func TestThinking(t *testing.T) {
	specs := map[string]openaicompat.Spec{
		"effort":    {Name: "test"},
		"always on": {Name: "test", ThinkingAlwaysOn: true},
		"type":      {Name: "test", Thinking: openaicompat.ThinkingType("enabled", true)},
		"type only": {Name: "test", Thinking: openaicompat.ThinkingType("adaptive", false)},
	}
	enabled := &litellm.Thinking{}
	high := &litellm.Thinking{Effort: "high"}
	budget := &litellm.Thinking{BudgetTokens: new(1024)}
	disabled := &litellm.Thinking{Mode: litellm.ThinkingDisabled}
	tests := []struct {
		spec     string
		thinking *litellm.Thinking
		want     string // fields, or the error text
	}{
		{"effort", enabled, `{"reasoning_effort": null}`},
		{"effort", high, `{"reasoning_effort": "high"}`},
		{"effort", disabled, `{"reasoning_effort": "none"}`},
		{"effort", budget, "budget_tokens is not supported"},
		{"always on", high, `{"reasoning_effort": "high"}`},
		{"always on", disabled, "thinking cannot be disabled"},
		{"type", enabled, `{"thinking": {"type": "enabled"}, "reasoning_effort": null}`},
		{"type", high, `{"thinking": {"type": "enabled"}, "reasoning_effort": "high"}`},
		{"type", disabled, `{"thinking": {"type": "disabled"}, "reasoning_effort": null}`},
		{"type", budget, "budget_tokens is not supported"},
		{"type only", enabled, `{"thinking": {"type": "adaptive"}}`},
		{"type only", high, "effort is not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.spec+"/"+tt.want, func(t *testing.T) {
			newFn := compattest.Spec(specs[tt.spec])
			req := compattest.Request()
			req.Thinking = tt.thinking
			if !strings.HasPrefix(tt.want, "{") {
				if err := compattest.Err(t, newFn, req); !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("err = %v, want %q", err, tt.want)
				}
				return
			}
			compattest.AssertFields(t, compattest.Body(t, newFn, req, false), tt.want)
		})
	}
}

func TestFieldsAreNotShared(t *testing.T) {
	newFn := compattest.Spec(openaicompat.Spec{Name: "test", Fields: map[string]any{"extra": map[string]any{"a": 1}}, Options: []string{"extra"}})
	req := compattest.Request()
	req.ProviderOptions = compattest.Options(t, map[string]any{"extra": map[string]any{"b": 2}})
	compattest.AssertFields(t, compattest.Body(t, newFn, req, false), `{"extra": {"a": 1, "b": 2}}`)
	compattest.AssertFields(t, compattest.Body(t, newFn, compattest.Request(), false), `{"extra": {"a": 1}}`)
}

func TestCacheBreakpoints(t *testing.T) {
	cache := &litellm.CacheControl{TTL: litellm.CacheTTL5m}
	messages := []litellm.Message{
		litellm.User(litellm.TextBlock{Text: "hi", Cache: cache}, litellm.ImageBlock{URL: "https://img.test/a.png", Cache: cache}),
		litellm.User(litellm.TextBlock{Text: "only", Cache: cache}),
		{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: "call_1", Content: []litellm.Block{litellm.Text("ok")}, Cache: cache}}},
	}
	mark := func(c *litellm.CacheControl) (map[string]any, error) {
		return map[string]any{"cache_control": map[string]any{"ttl": c.TTL}}, nil
	}
	tests := []struct {
		name  string
		cache func(*litellm.CacheControl) (map[string]any, error)
		want  string
	}{
		{name: "dropped without hook", want: `[
			{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "https://img.test/a.png"}}]},
			{"role": "user", "content": "only"},
			{"role": "tool", "tool_call_id": "call_1", "content": "ok"}
		]`},
		{name: "hook", cache: mark, want: `[
			{"role": "user", "content": [
				{"type": "text", "text": "hi", "cache_control": {"ttl": "5m"}},
				{"type": "image_url", "image_url": {"url": "https://img.test/a.png"}, "cache_control": {"ttl": "5m"}}
			]},
			{"role": "user", "content": [{"type": "text", "text": "only", "cache_control": {"ttl": "5m"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "ok", "cache_control": {"ttl": "5m"}}]}
		]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newFn := compattest.Spec(openaicompat.Spec{Name: "test", Cache: tt.cache})
			body := compattest.Body(t, newFn, &litellm.Request{Model: "m", Messages: messages}, false)
			compattest.AssertJSON(t, body["messages"], tt.want)
		})
	}
}

func TestAssistantMessage(t *testing.T) {
	state := func(provider, data string) *litellm.ProviderState {
		return &litellm.ProviderState{Provider: provider, Data: json.RawMessage(data)}
	}
	details := litellm.ReasoningBlock{Text: "t", State: state("test", `[{"type":"reasoning.text","text":"t"}]`)}
	call := litellm.ToolUseBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{}`)}
	toolCalls := `[{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]`
	both := []string{"reasoning_details", "reasoning"}
	tests := []struct {
		name   string
		spec   openaicompat.Spec
		blocks []litellm.Block
		want   string
	}{
		{name: "tool call without content", blocks: []litellm.Block{call}, want: `{"role": "assistant", "tool_calls": ` + toolCalls + `}`},
		{name: "empty tool call content", spec: openaicompat.Spec{StringContentRoles: []litellm.Role{litellm.RoleAssistant}}, blocks: []litellm.Block{call}, want: `{"role": "assistant", "content": "", "tool_calls": ` + toolCalls + `}`},
		{name: "reasoning dropped without fields", blocks: []litellm.Block{details, litellm.Text("ok")}, want: `{"role": "assistant", "content": "ok"}`},
		{name: "reasoning text", spec: openaicompat.Spec{ReasoningFields: []string{"reasoning_content"}}, blocks: []litellm.Block{litellm.ReasoningBlock{Text: "a"}, litellm.ReasoningBlock{Text: "b"}}, want: `{"role": "assistant", "reasoning_content": "a\n\nb"}`},
		{name: "state ignored without details field", spec: openaicompat.Spec{ReasoningFields: []string{"reasoning_content"}}, blocks: []litellm.Block{details}, want: `{"role": "assistant", "reasoning_content": "t"}`},
		{name: "text goes to text field", spec: openaicompat.Spec{ReasoningFields: []string{"reasoning_details", "reasoning_content"}}, blocks: []litellm.Block{litellm.ReasoningBlock{Text: "t"}}, want: `{"role": "assistant", "reasoning_content": "t"}`},
		{name: "details dropped without text field", spec: openaicompat.Spec{ReasoningFields: []string{"reasoning_details"}}, blocks: []litellm.Block{litellm.ReasoningBlock{Text: "t"}, litellm.Text("ok")}, want: `{"role": "assistant", "content": "ok"}`},
		{name: "details replayed", spec: openaicompat.Spec{ReasoningFields: both}, blocks: []litellm.Block{details}, want: `{"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "text": "t"}]}`},
		{
			// Details from another provider are not replayed, even in the same format.
			name: "foreign state falls back to text", spec: openaicompat.Spec{ReasoningFields: both},
			blocks: []litellm.Block{litellm.ReasoningBlock{Text: "t", State: state("other", `[{"type":"reasoning.text","text":"t"}]`)}},
			want:   `{"role": "assistant", "reasoning": "t"}`,
		},
		{
			name: "details supersede text", spec: openaicompat.Spec{ReasoningFields: both},
			blocks: []litellm.Block{litellm.ReasoningBlock{Text: "a"}, details, litellm.ReasoningBlock{State: state("test", `[{"type":"reasoning.encrypted","data":"x"}]`)}, litellm.ReasoningBlock{Text: "b"}},
			want:   `{"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "text": "t"}, {"type": "reasoning.encrypted", "data": "x"}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.spec.Name = "test"
			body := compattest.Body(t, compattest.Spec(tt.spec), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.Assistant(tt.blocks...)}}, false)
			compattest.AssertJSON(t, body["messages"].([]any)[0], tt.want)
		})
	}
}

// A message left with nothing to send, such as reasoning no field can carry,
// is omitted rather than sent empty.
func TestEmptyMessageOmitted(t *testing.T) {
	messages := []litellm.Message{litellm.UserText("a"), litellm.Assistant(litellm.ReasoningBlock{Text: "t"}), litellm.UserText("b")}
	body := compattest.Body(t, compattest.Spec(openaicompat.Spec{Name: "test"}), &litellm.Request{Model: "m", Messages: messages}, false)
	compattest.AssertJSON(t, body["messages"], `[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]`)
}

func TestNewAndHeaders(t *testing.T) {
	if _, err := openaicompat.New(openaicompat.Config{}, openaicompat.Spec{Name: "test"}); err == nil || !strings.Contains(err.Error(), "base url is required") {
		t.Fatalf("missing base url: %v", err)
	}
	if _, err := openaicompat.New(openaicompat.Config{BaseURL: "https://api.test"}, openaicompat.Spec{Name: "test", APIKeyRequired: true}); err == nil || !strings.Contains(err.Error(), "api key is required") {
		t.Fatalf("missing api key: %v", err)
	}
	var got []*http.Request
	p, err := openaicompat.New(openaicompat.Config{
		BaseURL:    "https://api.test/v1/",
		APIKeyFunc: func(context.Context) (string, error) { return "fn-key", nil },
		Headers:    map[string]string{"X-Team": "a"},
		HTTPClient: compattest.Doer(func(r *http.Request) (*http.Response, error) {
			got = append(got, r)
			if r.Header.Get("Accept") == "text/event-stream" {
				return compattest.Response(compattest.SSE()), nil
			}
			return compattest.Response(`{"choices":[]}`), nil
		}),
	}, openaicompat.Spec{Name: "test", APIKeyRequired: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), compattest.Request()); err != nil {
		t.Fatal(err)
	}
	s, err := p.Stream(context.Background(), compattest.Request())
	if err != nil {
		t.Fatal(err)
	}
	s.Close()
	for i, accept := range []string{"application/json", "text/event-stream"} {
		r := got[i]
		if r.URL.String() != "https://api.test/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer fn-key" ||
			r.Header.Get("User-Agent") != wire.DefaultUserAgent || r.Header.Get("X-Team") != "a" || r.Header.Get("Accept") != accept {
			t.Fatalf("request %d: %s %v", i, r.URL, r.Header)
		}
	}

	unkeyed := compattest.Spec(openaicompat.Spec{Name: "test"})
	p, err = unkeyed(openaicompat.Config{BaseURL: "https://api.test", HTTPClient: compattest.Doer(func(r *http.Request) (*http.Response, error) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Fatalf("Authorization = %q", auth)
		}
		return compattest.Response(`{"choices":[]}`), nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), compattest.Request()); err != nil {
		t.Fatal(err)
	}
}
