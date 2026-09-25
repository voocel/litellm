package bedrock

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
)

func TestBuildRequestGolden(t *testing.T) {
	hour := &litellm.CacheControl{TTL: litellm.CacheTTL1h}
	data, err := buildRequest(&litellm.Request{
		Model:       "anthropic.claude-opus-5",
		MaxTokens:   new(4096),
		Temperature: new(1.0),
		Messages: []litellm.Message{
			{Role: litellm.RoleSystem, Blocks: []litellm.Block{litellm.TextBlock{Text: "be concise", Cache: hour}}},
			litellm.UserText("use tool"),
			litellm.Assistant(litellm.ToolUseBlock{ID: "toolu_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)}),
			{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: "toolu_1", Content: []litellm.Block{litellm.Text("result")}, Cache: hour}}},
		},
		Tools: []litellm.Tool{{
			Name: "lookup", Description: "Lookup data.", Strict: litellm.StrictEnabled,
			Parameters: litellm.Schema(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
		Thinking: &litellm.Thinking{Effort: "low"},
	})
	if err != nil {
		t.Fatal(err)
	}
	testgolden.AssertJSONBytes(t, "../../testdata/bedrock/request_converse_tools.golden.json", data)
}

func TestBuildRequest(t *testing.T) {
	const schema = `{"type":"object","properties":{"q":{"type":"string"}}}`
	toolHistory := []litellm.Message{
		litellm.UserText("hi"),
		litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Name: "f"}),
		litellm.ToolResultText("t1", "one"),
	}
	for _, test := range []struct {
		name string
		req  func(*litellm.Request)
		// want maps body fields to their JSON; "" asserts the field is absent.
		want    map[string]string
		wantErr string
	}{
		{
			name: "no sampling or thinking sends neither",
			req:  func(*litellm.Request) {},
			want: map[string]string{"inferenceConfig": "", "additionalModelRequestFields": "", "toolConfig": ""},
		},
		{
			name: "inference config",
			req: func(r *litellm.Request) {
				r.MaxTokens, r.Temperature, r.TopP, r.Stop = new(10), new(0.5), new(0.9), []string{"END"}
			},
			want: map[string]string{"inferenceConfig": `{"maxTokens":10,"temperature":0.5,"topP":0.9,"stopSequences":["END"]}`},
		},
		{
			name: "thinking disabled",
			req:  func(r *litellm.Request) { r.Thinking = &litellm.Thinking{Mode: litellm.ThinkingDisabled} },
			want: map[string]string{"additionalModelRequestFields": `{"thinking":{"type":"disabled"}}`},
		},
		{
			name: "thinking budget",
			req:  func(r *litellm.Request) { r.Thinking = &litellm.Thinking{BudgetTokens: new(2048), IncludeOutput: true} },
			want: map[string]string{"additionalModelRequestFields": `{"thinking":{"type":"enabled","budget_tokens":2048,"display":"summarized"}}`},
		},
		{
			name: "additional fields option merges with thinking",
			req: func(r *litellm.Request) {
				r.Thinking = &litellm.Thinking{Effort: "high"}
				withOptions(r, map[string]any{ProviderOptionAdditionalModelRequestFields: map[string]any{
					"top_k": 5, "output_config": map[string]any{"task_budget": 100},
				}})
			},
			want: map[string]string{"additionalModelRequestFields": `{"thinking":{"type":"adaptive"},"output_config":{"effort":"high","task_budget":100},"top_k":5}`},
		},
		{
			name: "additional fields option colliding with thinking",
			req: func(r *litellm.Request) {
				r.Thinking = &litellm.Thinking{Effort: "high"}
				withOptions(r, map[string]any{ProviderOptionAdditionalModelRequestFields: map[string]any{"output_config": map[string]any{"effort": "low"}}})
			},
			wantErr: `provider option "additionalModelRequestFields.output_config.effort" conflicts with a generated request field`,
		},
		{
			name: "native options",
			req: func(r *litellm.Request) {
				withOptions(r, map[string]any{
					ProviderOptionGuardrailConfig:   map[string]any{"guardrailIdentifier": "g", "guardrailVersion": "1"},
					ProviderOptionPerformanceConfig: map[string]any{"latency": "optimized"},
				})
			},
			want: map[string]string{
				"guardrailConfig":   `{"guardrailIdentifier":"g","guardrailVersion":"1"}`,
				"performanceConfig": `{"latency":"optimized"}`,
			},
		},
		{
			name:    "unknown option",
			req:     func(r *litellm.Request) { withOptions(r, map[string]any{"thinking": true}) },
			wantErr: `unsupported provider option "thinking"`,
		},
		{
			name: "tools and choices",
			req: func(r *litellm.Request) {
				r.Tools = []litellm.Tool{{Name: "a", Parameters: litellm.Schema(schema), Strict: litellm.StrictDisabled}, {Name: "b"}}
				r.ToolChoice = &litellm.ToolChoice{Name: "a"}
			},
			want: map[string]string{"toolConfig": `{"tools":[
				{"toolSpec":{"name":"a","strict":false,"inputSchema":{"json":` + schema + `}}},
				{"toolSpec":{"name":"b","inputSchema":{"json":{"type":"object"}}}}],
				"toolChoice":{"tool":{"name":"a"}}}`},
		},
		{
			name: "required choice",
			req: func(r *litellm.Request) {
				r.Tools, r.ToolChoice = []litellm.Tool{{Name: "a"}}, &litellm.ToolChoice{Mode: litellm.ToolChoiceRequired}
			},
			want: map[string]string{"toolConfig": `{"tools":[{"toolSpec":{"name":"a","inputSchema":{"json":{"type":"object"}}}}],"toolChoice":{"any":{}}}`},
		},
		{
			name: "choice none omits tools without tool history",
			req: func(r *litellm.Request) {
				r.Tools, r.ToolChoice = []litellm.Tool{{Name: "a"}}, &litellm.ToolChoice{Mode: litellm.ToolChoiceNone}
			},
			want: map[string]string{"toolConfig": ""},
		},
		{
			name: "choice none with tool history",
			req: func(r *litellm.Request) {
				r.Messages = toolHistory
				r.Tools, r.ToolChoice = []litellm.Tool{{Name: "f"}}, &litellm.ToolChoice{Mode: litellm.ToolChoiceNone}
			},
			wantErr: "tool_choice none cannot be expressed when history contains tool calls",
		},
		{
			name: "same roles merge and tool results share a user turn",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{
					litellm.UserText("a"), litellm.UserText("b"),
					litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Name: "f"}, litellm.ToolUseBlock{ID: "t2", Name: "f"}),
					litellm.ToolResultText("t1", "one"),
					{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{ToolUseID: "t2", IsError: true, Content: []litellm.Block{litellm.Text("boom")}}}},
				}
			},
			want: map[string]string{"messages": `[
				{"role":"user","content":[{"text":"a"},{"text":"b"}]},
				{"role":"assistant","content":[{"toolUse":{"toolUseId":"t1","name":"f","input":{}}},{"toolUse":{"toolUseId":"t2","name":"f","input":{}}}]},
				{"role":"user","content":[
					{"toolResult":{"toolUseId":"t1","content":[{"text":"one"}]}},
					{"toolResult":{"toolUseId":"t2","content":[{"text":"boom"}],"status":"error"}}]}]`},
		},
		{
			name: "cache points follow their blocks",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.User(litellm.TextBlock{Text: "doc", Cache: &litellm.CacheControl{}}, litellm.Text("question"))}
			},
			want: map[string]string{"messages": `[{"role":"user","content":[{"text":"doc"},{"cachePoint":{"type":"default"}},{"text":"question"}]}]`},
		},
		{
			name: "images are sent as bytes",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.User(
					litellm.ImageBlock{Data: []byte("png"), MIME: "image/png"},
					litellm.ImageURL("data:image/jpeg;base64,anBn"),
				)}
			},
			want: map[string]string{"messages": `[{"role":"user","content":[
				{"image":{"format":"png","source":{"bytes":"cG5n"}}},
				{"image":{"format":"jpeg","source":{"bytes":"anBn"}}}]}]`},
		},
		{
			name: "remote image URL",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.User(litellm.ImageURL("https://x.test/a.png"))}
			},
			wantErr: "messages[0]: image requires inline data or a data URL",
		},
		{
			name: "image MIME must be an image type",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.User(litellm.ImageBlock{Data: []byte("x"), MIME: "text/plain"})}
			},
			wantErr: `messages[0]: image MIME "text/plain" must be image/<format>`,
		},
		{
			name: "reasoning history replays signature and redacted data",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.UserText("hi"), litellm.Assistant(
					litellm.ReasoningBlock{Text: "t", Signature: "sig"},
					litellm.ReasoningBlock{Redacted: []byte("opaque")},
					litellm.Text("answer"),
				)}
			},
			want: map[string]string{"messages": `[
				{"role":"user","content":[{"text":"hi"}]},
				{"role":"assistant","content":[
					{"reasoningContent":{"reasoningText":{"text":"t","signature":"sig"}}},
					{"reasoningContent":{"redactedContent":"b3BhcXVl"}},
					{"text":"answer"}]}]`},
		},
		{
			name: "tool arguments must be an object",
			req: func(r *litellm.Request) {
				r.Messages = []litellm.Message{litellm.Assistant(litellm.ToolUseBlock{ID: "t1", Arguments: json.RawMessage(`[1]`)})}
			},
			wantErr: `messages[0]: tool use "t1" arguments must be a JSON object`,
		},
		{
			name: "json schema output",
			req: func(r *litellm.Request) {
				r.ResponseFormat = &litellm.ResponseFormat{Type: litellm.ResponseFormatJSONSchema, JSONSchema: &litellm.JSONSchema{Name: "out", Schema: litellm.Schema(schema)}}
			},
			want: map[string]string{"outputConfig": `{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"name":"out","schema":` + jsonString(schema) + `}}}}`},
		},
		{
			name:    "json object output",
			req:     func(r *litellm.Request) { r.ResponseFormat = litellm.NewResponseFormatJSONObject() },
			wantErr: "response_format json_object has no Converse equivalent; use json_schema",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}}
			test.req(req)
			data, err := buildRequest(req)
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("err = %v, want %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
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

func withOptions(r *litellm.Request, values map[string]any) {
	options, err := litellm.NewProviderOptions(values)
	if err != nil {
		panic(err)
	}
	r.ProviderOptions = options
}

func jsonString(s string) string {
	data, _ := json.Marshal(s)
	return string(data)
}

func jsonEqual(t *testing.T, got json.RawMessage, want string) bool {
	t.Helper()
	var a, b any
	if err := json.Unmarshal(got, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &b); err != nil {
		t.Fatalf("want is not JSON: %v", err)
	}
	return reflect.DeepEqual(a, b)
}
