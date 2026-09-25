package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestBuildRequestGolden(t *testing.T) {
	data, err := buildRequest(&litellm.Request{
		Model:       "claude-sonnet-5",
		MaxTokens:   new(4096),
		Temperature: new(1.0),
		Messages: []litellm.Message{
			litellm.System("You are helpful."),
			litellm.User(litellm.TextBlock{Text: "Use the tool.", Cache: &litellm.CacheControl{TTL: litellm.CacheTTL1h}}),
			litellm.Assistant(
				litellm.ReasoningBlock{Text: "I should call the tool.", Signature: "sig-thinking"},
				litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
			),
			litellm.ToolResult("toolu_1", litellm.Text("result text"), litellm.ToolReferenceBlock{ToolName: "lookup"}),
			litellm.AssistantText("done"),
		},
		Tools: []litellm.Tool{{
			Name:        "lookup",
			Description: "Lookup data.",
			Parameters:  litellm.Schema(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
		Thinking: &litellm.Thinking{Effort: "low"},
	}, false)
	if err != nil {
		t.Fatalf("buildRequest: %v", err)
	}
	testgolden.AssertJSONBytes(t, "../../testdata/anthropic/request_tools_cache.golden.json", data)
}

func TestBuildRequest(t *testing.T) {
	const schema = `{"type":"object","properties":{"q":{"type":"string"}}}`
	jsonSchema := &litellm.ResponseFormat{Type: litellm.ResponseFormatJSONSchema, JSONSchema: &litellm.JSONSchema{Name: "out", Schema: litellm.Schema(schema)}}
	for _, test := range []struct {
		name string
		req  func(*litellm.Request)
		// want maps body fields to their JSON; "" asserts the field is absent.
		want    map[string]string
		wantErr string
	}{
		{
			name: "system text is a string",
			req:  withMessages(litellm.System("be brief"), litellm.UserText("hi")),
			want: map[string]string{"system": `"be brief"`},
		},
		{
			name: "system blocks are hoisted and keep cache",
			req: withMessages(litellm.UserText("hi"), litellm.Message{Role: litellm.RoleSystem, Blocks: []litellm.Block{
				litellm.Text("a"), litellm.TextBlock{Text: "b", Cache: &litellm.CacheControl{}},
			}}),
			want: map[string]string{
				"system":   `[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral"}}]`,
				"messages": `[{"role":"user","content":[{"type":"text","text":"hi"}]}]`,
			},
		},
		{
			name: "same roles merge",
			req:  withMessages(litellm.UserText("a"), litellm.UserText("b"), litellm.AssistantText("c"), litellm.AssistantText("d")),
			want: map[string]string{"messages": `[
				{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
				{"role":"assistant","content":[{"type":"text","text":"c"},{"type":"text","text":"d"}]}]`},
		},
		{
			name: "tool results share a user turn",
			req: withMessages(
				litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Name: "f"}, litellm.ToolUseBlock{ID: "t2", Name: "f", Arguments: json.RawMessage(`{}`)}),
				litellm.ToolResultText("t1", "one"),
				litellm.Message{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{
					ToolUseID: "t2", IsError: true, Content: []litellm.Block{litellm.TextBlock{Text: "boom", Cache: &litellm.CacheControl{TTL: litellm.CacheTTL5m}}},
				}}},
			),
			want: map[string]string{"messages": `[
				{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}},{"type":"tool_use","id":"t2","name":"f","input":{}}]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"t1","content":"one"},
					{"type":"tool_result","tool_use_id":"t2","is_error":true,"content":[{"type":"text","text":"boom","cache_control":{"type":"ephemeral","ttl":"5m"}}]}]}]`},
		},
		{
			name: "images",
			req:  withMessages(litellm.User(litellm.ImageURL("https://x.test/a.png"), litellm.ImageBlock{Data: []byte("png"), MIME: "image/png"})),
			want: map[string]string{"messages": `[{"role":"user","content":[
				{"type":"image","source":{"type":"url","url":"https://x.test/a.png"}},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"cG5n"}}]}]`},
		},
		{
			name:    "inline image without MIME",
			req:     withMessages(litellm.User(litellm.ImageBlock{Data: []byte("png")})),
			wantErr: "messages[0]: inline image requires MIME",
		},
		{
			name:    "image without source",
			req:     withMessages(litellm.User(litellm.ImageBlock{})),
			wantErr: "messages[0]: image requires URL or data",
		},
		{
			name:    "image file URI",
			req:     withMessages(litellm.User(litellm.ImageBlock{FileURI: "gs://b/a.png"})),
			wantErr: "messages[0]: image FileURI is not supported",
		},
		{
			name: "empty tool result omits content",
			req: withMessages(
				litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Name: "f"}),
				litellm.ToolResult("t1"),
			),
			want: map[string]string{"messages": `[
				{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}]`},
		},
		{
			name: "reasoning replays signature and redacted data",
			req: withMessages(litellm.UserText("hi"), litellm.Assistant(
				litellm.ReasoningBlock{Text: "t", Signature: "sig"},
				litellm.ReasoningBlock{Signature: "omitted"},
				litellm.ReasoningBlock{Redacted: []byte("opaque")},
				litellm.ToolUseBlock{ID: "t1", Name: "f", Arguments: json.RawMessage(`{"q":"x"}`)},
			)),
			want: map[string]string{"messages": `[
				{"role":"user","content":[{"type":"text","text":"hi"}]},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"t","signature":"sig"},
					{"type":"thinking","thinking":"","signature":"omitted"},
					{"type":"redacted_thinking","data":"opaque"},
					{"type":"tool_use","id":"t1","name":"f","input":{"q":"x"}}]}]`},
		},
		{
			name:    "tool arguments must be an object",
			req:     withMessages(litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Name: "f", Arguments: json.RawMessage(`[1]`)})),
			wantErr: `messages[0]: tool use "t1" arguments must be a JSON object`,
		},
		{
			name: "tools",
			req: func(r *litellm.Request) {
				r.Tools = []litellm.Tool{
					{Name: "a", Parameters: litellm.Schema(schema), Strict: litellm.StrictEnabled},
					{Name: "b", Description: "d", Strict: litellm.StrictDisabled},
				}
			},
			want: map[string]string{"tools": `[
				{"name":"a","input_schema":` + schema + `,"strict":true},
				{"name":"b","description":"d","input_schema":{"type":"object"},"strict":false}]`},
		},
		{name: "tool choice auto", req: withToolChoice(litellm.ToolChoice{Mode: litellm.ToolChoiceAuto}), want: map[string]string{"tool_choice": `{"type":"auto"}`}},
		{name: "tool choice required", req: withToolChoice(litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}), want: map[string]string{"tool_choice": `{"type":"any"}`}},
		{name: "tool choice none", req: withToolChoice(litellm.ToolChoice{Mode: litellm.ToolChoiceNone}), want: map[string]string{"tool_choice": `{"type":"none"}`}},
		{name: "tool choice name", req: withToolChoice(litellm.ToolChoice{Name: "a"}), want: map[string]string{"tool_choice": `{"type":"tool","name":"a"}`}},
		{
			name: "thinking unset",
			req:  func(*litellm.Request) {},
			want: map[string]string{"thinking": "", "output_config": ""},
		},
		{
			name: "thinking budget",
			req:  func(r *litellm.Request) { r.Thinking = &litellm.Thinking{BudgetTokens: new(2048)} },
			want: map[string]string{"thinking": `{"type":"enabled","budget_tokens":2048}`, "output_config": ""},
		},
		{
			name: "effort and json schema share output_config",
			req: func(r *litellm.Request) {
				r.Thinking = &litellm.Thinking{Effort: "high"}
				r.ResponseFormat = jsonSchema
			},
			want: map[string]string{
				"thinking":      `{"type":"adaptive"}`,
				"output_config": `{"effort":"high","format":{"type":"json_schema","schema":` + schema + `}}`,
			},
		},
		{
			name: "text format",
			req:  func(r *litellm.Request) { r.ResponseFormat = &litellm.ResponseFormat{Type: litellm.ResponseFormatText} },
			want: map[string]string{"output_config": ""},
		},
		{
			name: "json object",
			req: func(r *litellm.Request) {
				r.ResponseFormat = &litellm.ResponseFormat{Type: litellm.ResponseFormatJSONObject}
			},
			wantErr: "response_format json_object has no Messages API equivalent; use json_schema",
		},
		{
			name: "sampling is sent as given",
			req: func(r *litellm.Request) {
				r.Temperature, r.TopP, r.Stop = new(0.5), new(0.9), []string{"END"}
			},
			want: map[string]string{"temperature": `0.5`, "top_p": `0.9`, "stop_sequences": `["END"]`, "stream": ""},
		},
		{
			name:    "max_tokens is required",
			req:     func(r *litellm.Request) { r.MaxTokens = nil },
			wantErr: "max_tokens is required by the Messages API",
		},
		{
			name: "options",
			req: withOptions(map[string]any{
				ProviderOptionMetadata: map[string]any{"user_id": "u"}, ProviderOptionTopK: 5, ProviderOptionServiceTier: "auto",
			}),
			want: map[string]string{"metadata": `{"user_id":"u"}`, "top_k": `5`, "service_tier": `"auto"`},
		},
		{
			name: "tools option appends to generated tools",
			req: func(r *litellm.Request) {
				r.Tools = []litellm.Tool{{Name: "a"}}
				withOptions(map[string]any{ProviderOptionTools: []any{map[string]any{"type": "web_search_20250305", "name": "web_search"}}})(r)
			},
			want: map[string]string{"tools": `[{"name":"a","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}]`},
		},
		{
			name: "output_config option merges into generated output_config",
			req: func(r *litellm.Request) {
				r.ResponseFormat = jsonSchema
				withOptions(map[string]any{ProviderOptionOutputConfig: map[string]any{"effort": "medium"}})(r)
			},
			want: map[string]string{"output_config": `{"effort":"medium","format":{"type":"json_schema","schema":` + schema + `}}`},
		},
		{
			name: "tool_choice option merges into generated tool_choice",
			req: func(r *litellm.Request) {
				withToolChoice(litellm.ToolChoice{Mode: litellm.ToolChoiceAuto})(r)
				withOptions(map[string]any{ProviderOptionToolChoice: map[string]any{"disable_parallel_tool_use": true}})(r)
			},
			want: map[string]string{"tool_choice": `{"type":"auto","disable_parallel_tool_use":true}`},
		},
		{
			name:    "unknown option",
			req:     withOptions(map[string]any{"thinking": map[string]any{"type": "enabled"}}),
			wantErr: `unsupported provider option "thinking"`,
		},
		{
			name: "option conflicting with a generated field",
			req: func(r *litellm.Request) {
				r.Tools = []litellm.Tool{{Name: "a"}}
				withOptions(map[string]any{ProviderOptionTools: map[string]any{"name": "b"}})(r)
			},
			wantErr: `provider option "tools" conflicts with a generated request field`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := &litellm.Request{Model: "claude", MaxTokens: new(1024), Messages: []litellm.Message{litellm.UserText("hi")}}
			test.req(req)
			data, err := buildRequest(req, false)
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("err = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("buildRequest: %v", err)
			}
			var body map[string]json.RawMessage
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatal(err)
			}
			for key, want := range test.want {
				got, ok := body[key]
				if want == "" {
					if ok {
						t.Errorf("%s = %s, want absent", key, got)
					}
					continue
				}
				if !ok || !jsonEqual(t, got, want) {
					t.Errorf("%s = %s\nwant %s", key, got, want)
				}
			}
		})
	}
}

func withMessages(messages ...litellm.Message) func(*litellm.Request) {
	return func(r *litellm.Request) { r.Messages = messages }
}

func withToolChoice(choice litellm.ToolChoice) func(*litellm.Request) {
	return func(r *litellm.Request) { r.ToolChoice = &choice }
}

func withOptions(values map[string]any) func(*litellm.Request) {
	return func(r *litellm.Request) {
		options, err := litellm.NewProviderOptions(values)
		if err != nil {
			panic(err)
		}
		r.ProviderOptions = options
	}
}

func jsonEqual(t *testing.T, got json.RawMessage, want string) bool {
	t.Helper()
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("decode %s: %v", got, err)
	}
	if err := json.NewDecoder(strings.NewReader(want)).Decode(&w); err != nil {
		t.Fatalf("decode want %s: %v", want, err)
	}
	return reflect.DeepEqual(g, w)
}
