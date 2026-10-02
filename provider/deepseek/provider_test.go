package deepseek_test

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
)

func TestMessageFormats(t *testing.T) {
	req := &litellm.Request{Model: "deepseek-flash", Messages: []litellm.Message{
		{Role: litellm.RoleSystem, Blocks: []litellm.Block{litellm.Text("Be "), litellm.Text("brief.")}},
		litellm.User(litellm.Text("Look"), litellm.ImageBlock{FileURI: "file-api-image"},
			litellm.ImageBlock{URL: "https://example.com/image.png", Detail: "original"},
			litellm.ImageBlock{Data: []byte("png"), MIME: "image/png", Detail: "low"}),
		litellm.Assistant(litellm.ReasoningBlock{Text: "reasoning"}, litellm.Text("Hello "), litellm.Text("world")),
		litellm.Assistant(litellm.ReasoningBlock{Text: "unfinished reasoning"}),
	}}
	for _, streaming := range []bool{false, true} {
		body := compattest.Body(t, deepseek.New, req, streaming)
		compattest.AssertJSON(t, body["messages"], `[
			{"role":"system","content":"Be brief."},
			{"role":"user","content":[
				{"type":"text","text":"Look"},
				{"type":"file","file_id":"file-api-image"},
				{"type":"image_url","image_url":{"url":"https://example.com/image.png","detail":"original"}},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n","detail":"low"}}
			]},
			{"role":"assistant","content":"Hello world","reasoning_content":"reasoning"},
			{"role":"assistant","content":"","reasoning_content":"unfinished reasoning"}
		]`)
	}
}

func TestResponseFormats(t *testing.T) {
	for _, format := range []*litellm.ResponseFormat{nil, {}, litellm.NewResponseFormatText(), litellm.NewResponseFormatJSONObject()} {
		req := compattest.Request()
		req.ResponseFormat = format
		body := compattest.Body(t, deepseek.New, req, false)
		if format != nil && format.Type == litellm.ResponseFormatJSONObject {
			compattest.AssertJSON(t, body["response_format"], `{"type":"json_object"}`)
		} else if _, ok := body["response_format"]; ok {
			t.Fatal("default text format should be omitted")
		}
	}
}

func TestUnsupportedRequests(t *testing.T) {
	tests := []struct {
		name string
		req  litellm.Request
		want string
	}{
		{"system image", litellm.Request{Messages: []litellm.Message{{Role: litellm.RoleSystem, Blocks: []litellm.Block{litellm.ImageURL("https://example.com/image.png")}}}}, "system messages do not support images"},
		{"assistant file", litellm.Request{Messages: []litellm.Message{litellm.Assistant(litellm.ImageBlock{FileURI: "file-api-image"})}}, "assistant messages do not support images"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := compattest.Provider(t, deepseek.New, compattest.Doer(func(*http.Request) (*http.Response, error) {
				t.Fatal("invalid request was sent")
				return nil, nil
			}))
			tt.req.Model = "deepseek-flash"
			_, chatErr := p.Chat(context.Background(), &tt.req)
			_, streamErr := p.Stream(context.Background(), &tt.req)
			for _, err := range []error{chatErr, streamErr} {
				if litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("error = %v, want validation error containing %q", err, tt.want)
				}
			}
		})
	}
}

func TestReasoningToolRoundTrip(t *testing.T) {
	const complete = `{"choices":[{"message":{"content":"","reasoning_content":"先查日期。\n再查天气。","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_cache_hit_tokens":4}}`
	sse := compattest.SSE(
		`{"choices":[{"index":0,"delta":{"reasoning_content":"先查日期。\n"}}]}`,
		`{"choices":[{"index":0,"delta":{"reasoning_content":"再查天气。"}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather","arguments":"{"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`,
		`{"choices":[{"index":0,"delta":{"content":""},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_cache_hit_tokens":4}}`,
	)
	for _, mode := range []string{"chat", "stream"} {
		t.Run(mode, func(t *testing.T) {
			var resp *litellm.Response
			var err error
			if mode == "chat" {
				resp, err = compattest.Chat(t, deepseek.New, complete)
			} else {
				resp, err = compattest.Collect(t, deepseek.New, sse)
			}
			if err != nil {
				t.Fatal(err)
			}
			if resp.Reasoning() != "先查日期。\n再查天气。" {
				t.Fatalf("reasoning = %q", resp.Reasoning())
			}
			if resp.Usage.CacheReadTokens != 4 {
				t.Fatalf("usage = %+v", resp.Usage)
			}
			req := &litellm.Request{Model: "deepseek-flash", Tools: []litellm.Tool{{Name: "weather"}}, Messages: []litellm.Message{
				litellm.UserText("昨天呢？"),
				litellm.Assistant(litellm.ReasoningBlock{Text: "之前的思考。"}, litellm.Text("晴天")),
				litellm.UserText("今天呢？"), litellm.Assistant(resp.Blocks...), litellm.ToolResultText("call_1", "sunny"),
			}}
			for _, streaming := range []bool{false, true} {
				body := compattest.Body(t, deepseek.New, req, streaming)
				messages := body["messages"].([]any)
				compattest.AssertJSON(t, messages[1], `{"role":"assistant","content":"晴天","reasoning_content":"之前的思考。"}`)
				compattest.AssertJSON(t, messages[3], `{"role":"assistant","content":"","reasoning_content":"先查日期。\n再查天气。","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]}`)
				compattest.AssertJSON(t, messages[4], `{"role":"tool","tool_call_id":"call_1","content":"sunny"}`)
			}
		})
	}
}

func TestStrictBetaEndpoint(t *testing.T) {
	p, err := deepseek.New(deepseek.Config{APIKey: "test", BaseURL: "https://api.deepseek.com/beta", HTTPClient: compattest.Doer(func(r *http.Request) (*http.Response, error) {
		if r.URL.String() != "https://api.deepseek.com/beta/chat/completions" {
			t.Fatalf("URL = %s", r.URL)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		compattest.AssertJSON(t, body["tools"], `[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{},"additionalProperties":false},"strict":true}}]`)
		return compattest.Response(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`), nil
	})})
	if err != nil {
		t.Fatal(err)
	}
	req := compattest.Request()
	req.Tools = []litellm.Tool{{Name: "weather", Strict: new(true), Parameters: litellm.Schema(`{"type":"object","properties":{},"additionalProperties":false}`)}}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}
