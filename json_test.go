package litellm

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/voocel/litellm/internal/testgolden"
)

// history holds every block kind with every field set, so the golden file
// pins the stored format of each.
func history() []Message {
	state := &ProviderState{Provider: "anthropic", Model: "claude", Data: json.RawMessage(`{"signature":"sig"}`)}
	cache := &CacheControl{}
	return []Message{
		System("be brief"),
		User(
			TextBlock{Text: "look", Cache: cache},
			ImageBlock{Data: []byte{0x89, 'P', 'N', 'G'}, MIME: "image/png", Detail: "high"},
			ImageBlock{URL: "https://example.com/a.png"},
			ImageBlock{FileURI: "gs://bucket/a.png", MIME: "image/png"},
		),
		Assistant(
			ReasoningBlock{Text: "thinking", Summary: true, State: state},
			TextBlock{
				Text:        "see",
				Annotations: []Annotation{{Type: "url_citation", Text: "src", URL: "https://example.com", Extra: json.RawMessage(`{"start":1}`)}},
				Logprobs:    json.RawMessage(`[{"token":"see"}]`),
				State:       state,
			},
			ToolUseBlock{ID: "call_1", Name: "read", Arguments: `{"path":"a.go"}`, State: state, Cache: cache},
		),
		{Role: RoleTool, Blocks: []Block{ToolResultBlock{
			ToolUseID: "call_1",
			Content:   []Block{Text("package a"), ImageURL("https://example.com/b.png"), ToolReferenceBlock{ToolName: "grep"}},
			IsError:   true,
			Cache:     cache,
		}}},
	}
}

func TestMessageJSON(t *testing.T) {
	want := history()
	testgolden.AssertJSON(t, "testdata/messages.golden.json", want)

	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got []Message
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed the history:\ngot  %#v\nwant %#v", got, want)
	}
}

func TestUnmarshalBlockRejectsUnknownType(t *testing.T) {
	var msg Message
	err := json.Unmarshal([]byte(`{"role":"user","blocks":[{"type":"video"}]}`), &msg)
	if err == nil {
		t.Fatal("decoded a block of unknown type")
	}
}

func TestRequestJSON(t *testing.T) {
	maxTokens, budget := 512, 1024
	temperature := 0.5
	options, err := NewProviderOptions(map[string]any{"prompt_cache_key": "conv-1"})
	if err != nil {
		t.Fatal(err)
	}
	tool, err := NewTool("read", "Read a file", map[string]any{"type": "object"})
	if err != nil {
		t.Fatal(err)
	}
	want := Request{
		Model:           "claude",
		Messages:        history(),
		MaxTokens:       &maxTokens,
		Temperature:     &temperature,
		Stop:            []string{"END"},
		Tools:           []Tool{tool},
		ToolChoice:      &ToolChoice{Mode: ToolChoiceAuto},
		ResponseFormat:  &ResponseFormat{Type: ResponseFormatJSONSchema, JSONSchema: &JSONSchema{Name: "out", Schema: Schema(`{"type":"object"}`), Strict: new(true)}},
		Thinking:        &Thinking{Effort: "high", BudgetTokens: &budget},
		ProviderOptions: options,
	}
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got Request
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip changed the request:\ngot  %#v\nwant %#v", got, want)
	}
}

// Every form of JSON text is checked and copied; other values are marshaled.
func TestSchemaFrom(t *testing.T) {
	text := []byte(`{"type":"object"}`)
	for _, v := range []any{Schema(text), json.RawMessage(text), text, string(text), map[string]any{"type": "object"}} {
		s, err := SchemaFrom(v)
		if err != nil || string(s) != `{"type":"object"}` {
			t.Fatalf("%T: %s, %v", v, s, err)
		}
	}
	if s, _ := SchemaFrom(text); &s[0] == &text[0] {
		t.Fatal("the schema shares the caller's bytes")
	}
	for _, v := range []any{Schema(`{`), json.RawMessage(`{`), []byte(`{`), `{`} {
		if _, err := SchemaFrom(v); err == nil {
			t.Fatalf("%T: accepted invalid JSON", v)
		}
	}
	if s, err := SchemaFrom(nil); s != nil || err != nil {
		t.Fatalf("nil: %s, %v", s, err)
	}
}
