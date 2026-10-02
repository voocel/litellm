package openaicompat_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
)

func schemaRequest() *litellm.Request {
	req := compattest.Request()
	req.ResponseFormat = &litellm.ResponseFormat{
		Type: litellm.ResponseFormatJSONSchema,
		JSONSchema: &litellm.JSONSchema{
			Name: "answer", Description: "天气预报", Strict: new(true),
			Schema: litellm.Schema(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
		},
	}
	return req
}

func TestSchemaOutput(t *testing.T) {
	for _, w := range wrappers {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", w.name, streaming), func(t *testing.T) {
				fallback := w.name == "deepseek" || w.name == "glm" || w.name == "mimo" || w.name == "minimax"
				req := schemaRequest()
				before, err := json.Marshal(req)
				if err != nil {
					t.Fatal(err)
				}
				p := compattest.Provider(t, w.newFn, compattest.Doer(func(r *http.Request) (*http.Response, error) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					if fallback {
						if w.name == "minimax" {
							if _, ok := body["response_format"]; ok {
								t.Fatal("MiniMax must omit response_format")
							}
						} else {
							compattest.AssertJSON(t, body["response_format"], `{"type":"json_object"}`)
						}
						messages := body["messages"].([]any)
						if len(messages) != 1 {
							t.Fatalf("messages = %v", messages)
						}
						parts := messages[0].(map[string]any)["content"].([]any)
						if len(parts) != 2 || parts[0].(map[string]any)["text"] != "hi" {
							t.Fatalf("content = %v", parts)
						}
						assertSchemaPrompt(t, parts[1].(map[string]any)["text"].(string), req.ResponseFormat.JSONSchema)
					} else {
						compattest.AssertJSON(t, body["messages"], `[{"role":"user","content":"hi"}]`)
						format := body["response_format"].(map[string]any)
						if format["type"] != "json_schema" {
							t.Fatalf("format = %v", format)
						}
						schema := format["json_schema"].(map[string]any)
						if schema["name"] != "answer" || schema["description"] != "天气预报" || schema["strict"] != true {
							t.Fatalf("schema = %v", schema)
						}
						compattest.AssertJSON(t, schema["schema"], string(req.ResponseFormat.JSONSchema.Schema))
					}
					if streaming {
						return compattest.Response(compattest.SSE(`{"choices":[{"index":0,"delta":{"content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}]}`)), nil
					}
					return compattest.Response(`{"choices":[{"message":{"content":"{\"city\":\"Paris\"}"},"finish_reason":"stop"}]}`), nil
				}))
				// Reusing the same request must not accumulate prompt instructions.
				for range 2 {
					var resp *litellm.Response
					var err error
					if streaming {
						s, streamErr := p.Stream(context.Background(), req)
						if streamErr != nil {
							t.Fatal(streamErr)
						}
						resp, err = litellm.Collect(s)
						s.Close()
					} else {
						resp, err = p.Chat(context.Background(), req)
					}
					if err != nil {
						t.Fatal(err)
					}
					if resp.Text() != `{"city":"Paris"}` {
						t.Fatalf("text = %q", resp.Text())
					}
					if fallback {
						if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "litellm.schema_prompt_fallback" || resp.Warnings[0].Provider != w.name {
							t.Fatalf("warnings = %+v", resp.Warnings)
						}
					} else if len(resp.Warnings) != 0 {
						t.Fatalf("native schema warnings = %+v", resp.Warnings)
					}
					compattest.AssertJSON(t, req, string(before))
				}
			})
		}
	}
}

func TestSchemaPromptPreservesHistory(t *testing.T) {
	req := schemaRequest()
	blocks := []litellm.Block{litellm.Text("look"), litellm.ImageURL("https://example.com/image.png"), litellm.Text("spare capacity")}
	req.Messages = []litellm.Message{
		litellm.System("Be brief."),
		litellm.UserText("earlier"),
		litellm.Assistant(litellm.Text("OK")),
		litellm.User(blocks[:2]...),
		litellm.Assistant(litellm.ReasoningBlock{Text: "check weather"}, litellm.ToolUseBlock{ID: "call_1", Name: "weather", Arguments: `{}`}),
		litellm.ToolResultText("call_1", "sunny"),
	}
	req.Tools = []litellm.Tool{{Name: "weather", Parameters: litellm.Schema(`{"type":"object"}`)}}
	before, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	for _, streaming := range []bool{false, true} {
		body := compattest.Body(t, deepseek.New, req, streaming)
		messages := body["messages"].([]any)
		if len(messages) != 6 {
			t.Fatalf("messages = %v", messages)
		}
		compattest.AssertJSON(t, messages[:3], `[{"role":"system","content":"Be brief."},{"role":"user","content":"earlier"},{"role":"assistant","content":"OK"}]`)
		parts := messages[3].(map[string]any)["content"].([]any)
		if len(parts) != 3 {
			t.Fatalf("content = %v", parts)
		}
		compattest.AssertJSON(t, parts[:2], `[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]`)
		assertSchemaPrompt(t, parts[2].(map[string]any)["text"].(string), req.ResponseFormat.JSONSchema)
		compattest.AssertJSON(t, messages[4:], `[
			{"role":"assistant","content":"","reasoning_content":"check weather","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		]`)
		compattest.AssertJSON(t, req, string(before))
		if blocks[2].(litellm.TextBlock).Text != "spare capacity" {
			t.Fatal("schema prompt overwrote caller history or its backing array")
		}
	}
}

func TestSchemaPromptWithoutUser(t *testing.T) {
	req := schemaRequest()
	req.Messages = []litellm.Message{litellm.System("Describe Paris.")}
	body := compattest.Body(t, deepseek.New, req, false)
	messages := body["messages"].([]any)
	if len(messages) != 2 || len(req.Messages) != 1 {
		t.Fatalf("messages = %v, original = %v", messages, req.Messages)
	}
	compattest.AssertJSON(t, messages[0], `{"role":"system","content":"Describe Paris."}`)
	user := messages[1].(map[string]any)
	if user["role"] != "user" {
		t.Fatalf("message = %v", user)
	}
	assertSchemaPrompt(t, user["content"].(string), req.ResponseFormat.JSONSchema)
}

func assertSchemaPrompt(t *testing.T, prompt string, schema *litellm.JSONSchema) {
	t.Helper()
	for _, value := range []string{"JSON", schema.Name, schema.Description, string(schema.Schema)} {
		if !strings.Contains(prompt, value) {
			t.Fatalf("prompt %q missing %q", prompt, value)
		}
	}
}
