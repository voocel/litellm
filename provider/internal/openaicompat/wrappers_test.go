package openaicompat_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/testgolden"
	"github.com/voocel/litellm/provider/compat"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/glm"
	"github.com/voocel/litellm/provider/grok"
	"github.com/voocel/litellm/provider/internal/openaicompat/compattest"
	"github.com/voocel/litellm/provider/mimo"
	"github.com/voocel/litellm/provider/minimax"
	"github.com/voocel/litellm/provider/ollama"
	"github.com/voocel/litellm/provider/openrouter"
	"github.com/voocel/litellm/provider/qwen"
)

// wrappers lists every Chat Completions dialect with its static facts and a
// representative request, encoded in testdata/compat/<name>_request.golden.json.
var wrappers = []struct {
	name    string
	newFn   compattest.NewFunc
	caps    litellm.Capabilities
	request func(*litellm.Request)
}{
	{name: "compat", newFn: compat.New, caps: litellm.Capabilities{ThinkingEffort: true, DisableThinking: true}},
	{
		name: "deepseek", newFn: deepseek.New,
		caps: litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{"frequency_penalty", "logprobs", "presence_penalty", "top_logprobs", "user_id"}},
		request: func(r *litellm.Request) {
			r.Model, r.Thinking = "deepseek-flash", &litellm.Thinking{Effort: "max"}
			r.Tools = []litellm.Tool{{Name: "lookup", Description: "Lookup.", Strict: new(true)}}
		},
	},
	{
		name: "glm", newFn: glm.New,
		caps:    litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{"do_sample", "request_id", "thinking", "tool_stream", "user_id"}},
		request: func(r *litellm.Request) { r.Model, r.Thinking = "glm-5.2", &litellm.Thinking{Effort: "max"} },
	},
	{
		name: "grok", newFn: grok.New,
		caps:    litellm.Capabilities{ThinkingEffort: true, ProviderOptions: []string{"frequency_penalty", "logprobs", "presence_penalty", "top_logprobs", "user"}},
		request: func(r *litellm.Request) { r.Model, r.Thinking = "grok-4.3", &litellm.Thinking{Effort: "high"} },
	},
	{
		name: "mimo", newFn: mimo.New,
		caps: litellm.Capabilities{DisableThinking: true, ProviderOptions: []string{"frequency_penalty", "presence_penalty"}},
		request: func(r *litellm.Request) {
			r.Model, r.MaxTokens, r.Thinking = "mimo-v2.5-pro", new(2048), &litellm.Thinking{Disabled: true}
		},
	},
	{
		name: "minimax", newFn: minimax.New,
		caps: litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{"service_tier"}},
		request: func(r *litellm.Request) {
			r.Model, r.MaxTokens, r.Thinking = "MiniMax-M3", new(128), &litellm.Thinking{}
		},
	},
	{
		name: "ollama", newFn: ollama.New,
		caps:    litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{"frequency_penalty", "presence_penalty", "seed"}},
		request: func(r *litellm.Request) { r.Model, r.Thinking = "qwen3", &litellm.Thinking{Effort: "high"} },
	},
	{
		name: "openrouter", newFn: openrouter.New,
		caps: litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{"cache_control", "provider", "session_id"}},
		request: func(r *litellm.Request) {
			r.Model, r.Thinking = "anthropic/claude-sonnet-4", &litellm.Thinking{Effort: "high"}
			r.Messages = []litellm.Message{litellm.User(litellm.TextBlock{Text: "hi", Cache: &litellm.CacheControl{}})}
			r.ProviderOptions = map[string]json.RawMessage{"cache_control": json.RawMessage(`{"type":"ephemeral"}`)}
		},
	},
	{
		name: "qwen", newFn: qwen.New,
		caps: litellm.Capabilities{DisableThinking: true, ProviderOptions: []string{
			"enable_search", "logprobs", "parallel_tool_calls", "presence_penalty",
			"preserve_thinking", "repetition_penalty", "search_options", "seed", "skill", "tool_stream", "top_k", "top_logprobs", "vl_high_resolution_images",
		}},
		request: func(r *litellm.Request) {
			r.Model, r.MaxTokens, r.Thinking = "qwen3.7-plus", new(8192), &litellm.Thinking{BudgetTokens: new(4096)}
		},
	},
}

func TestWrappers(t *testing.T) {
	for _, w := range wrappers {
		t.Run(w.name, func(t *testing.T) {
			p := compattest.Provider(t, w.newFn, nil)
			if p.Name() != w.name {
				t.Fatalf("Name = %q", p.Name())
			}
			if got := p.Capabilities(); !reflect.DeepEqual(got, w.caps) {
				t.Fatalf("Capabilities = %+v\nwant %+v", got, w.caps)
			}
			if w.request != nil {
				req := compattest.Request()
				w.request(req)
				testgolden.AssertJSON(t, "../../../testdata/compat/"+w.name+"_request.golden.json", compattest.Body(t, w.newFn, req, false))
			}
			// Every dialect reads reasoning_content, so the shared fixture holds.
			got, err := compattest.Collect(t, w.newFn, testgolden.ReadFixtureString(t, "../../../testdata/compat/stream.sse"))
			if err != nil {
				t.Fatal(err)
			}
			want, err := compattest.Chat(t, w.newFn, streamComplete)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("stream   %#v\ncomplete %#v", got, want)
			}
		})
	}
}

func TestWrapperThinking(t *testing.T) {
	enabled := &litellm.Thinking{}
	high := &litellm.Thinking{Effort: "high"}
	budget := &litellm.Thinking{BudgetTokens: new(1024)}
	disabled := &litellm.Thinking{Disabled: true}
	tests := []struct {
		name     string
		newFn    compattest.NewFunc
		thinking *litellm.Thinking
		want     string // fields, or the error text
	}{
		{"compat disabled", compat.New, disabled, `{"reasoning_effort": "none"}`},
		{"deepseek budget", deepseek.New, budget, "budget_tokens is not supported"},
		{"glm disabled", glm.New, disabled, `{"thinking": {"type": "disabled"}, "reasoning_effort": null}`},
		{"grok disabled", grok.New, disabled, "thinking cannot be disabled"},
		{"mimo enabled", mimo.New, enabled, `{"thinking": {"type": "enabled"}}`},
		{"mimo effort", mimo.New, high, "effort is not supported"},
		{"minimax effort", minimax.New, high, `{"thinking": {"type": "adaptive"}, "reasoning_effort": "high"}`},
		{"ollama disabled", ollama.New, disabled, `{"reasoning_effort": "none"}`},
		{"openrouter enabled", openrouter.New, enabled, `{"reasoning": {"enabled": true}}`},
		{"openrouter budget", openrouter.New, budget, `{"reasoning": {"max_tokens": 1024}}`},
		{"openrouter effort and budget", openrouter.New, &litellm.Thinking{Effort: "low", BudgetTokens: new(1024)}, "effort and budget_tokens cannot be combined"},
		{"openrouter disabled", openrouter.New, disabled, `{"reasoning": {"effort": "none"}}`},
		{"qwen enabled", qwen.New, enabled, `{"enable_thinking": true, "thinking_budget": null}`},
		{"qwen disabled", qwen.New, disabled, `{"enable_thinking": false}`},
		{"qwen effort", qwen.New, high, "thinking effort is not supported; use budget_tokens"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := compattest.Request()
			req.Thinking = tt.thinking
			if !strings.HasPrefix(tt.want, "{") {
				if err := compattest.Err(t, tt.newFn, req); litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("err = %v, want %q", err, tt.want)
				}
				return
			}
			compattest.AssertFields(t, compattest.Body(t, tt.newFn, req, false), tt.want)
		})
	}
}

func TestWrapperDialects(t *testing.T) {
	call := litellm.Assistant(litellm.ToolUseBlock{ID: "call_1", Name: "f", Arguments: `{}`})
	t.Run("compat passes unknown options and needs no key", func(t *testing.T) {
		req := compattest.Request()
		req.ProviderOptions = compattest.Options(t, map[string]any{"min_p": 0.05})
		compattest.AssertFields(t, compattest.Body(t, compat.New, req, false), `{"min_p": 0.05}`)
		if _, err := compat.New(compat.Config{BaseURL: "http://localhost:8000/v1"}); err != nil {
			t.Fatal(err)
		}
		if _, err := compat.New(compat.Config{}); err == nil || !strings.Contains(err.Error(), "base url is required") {
			t.Fatalf("missing base url: %v", err)
		}
	})
	t.Run("vendors require a key and reject unknown options", func(t *testing.T) {
		if _, err := deepseek.New(deepseek.Config{}); err == nil || !strings.Contains(err.Error(), "api key is required") {
			t.Fatalf("missing key: %v", err)
		}
		if _, err := ollama.New(ollama.Config{}); err != nil {
			t.Fatalf("ollama needs no key: %v", err)
		}
		req := compattest.Request()
		req.ProviderOptions = compattest.Options(t, map[string]any{"min_p": 0.05})
		if err := compattest.Err(t, grok.New, req); !strings.Contains(err.Error(), `unsupported provider option "min_p"`) {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("deepseek sends empty content with tool calls", func(t *testing.T) {
		body := compattest.Body(t, deepseek.New, &litellm.Request{Model: "m", Messages: []litellm.Message{call}}, false)
		compattest.AssertJSON(t, body["messages"].([]any)[0].(map[string]any)["content"], `""`)
	})
	t.Run("glm thinking option merges", func(t *testing.T) {
		req := compattest.Request()
		req.Thinking = &litellm.Thinking{}
		req.ProviderOptions = compattest.Options(t, map[string]any{glm.ProviderOptionThinking: map[string]any{"clear_thinking": false}})
		compattest.AssertFields(t, compattest.Body(t, glm.New, req, false), `{"thinking": {"type": "enabled", "clear_thinking": false}}`)
	})
	t.Run("mimo omits stream options", func(t *testing.T) {
		compattest.AssertFields(t, compattest.Body(t, mimo.New, compattest.Request(), true), `{"stream": true, "stream_options": null}`)
	})
	t.Run("minimax stream", func(t *testing.T) {
		// Incremental deltas, one reasoning_details entry per run, no [DONE].
		got, err := compattest.Collect(t, minimax.New, testgolden.ReadFixtureString(t, "../../../testdata/compat/minimax_stream.sse"))
		if err != nil {
			t.Fatal(err)
		}
		reasoning, _ := got.Blocks[0].(litellm.ReasoningBlock)
		want := `[{"format":"MiniMax-response-v1","id":"reasoning-text-1","index":0,"text":"ab","type":"reasoning.text"}]`
		if reasoning.Text != "ab" || reasoning.State.Provider != "minimax" || string(reasoning.State.Data) != want || got.Text() != "hi" || got.FinishReason != litellm.FinishReasonToolCall {
			t.Fatalf("response = %#v", got)
		}
	})
	t.Run("openrouter replays reasoning details", func(t *testing.T) {
		state := &litellm.ProviderState{Provider: "openrouter", Data: json.RawMessage(`[{"type":"reasoning.encrypted","data":"x"}]`)}
		msg := litellm.Assistant(litellm.ReasoningBlock{Text: "t", State: state}, litellm.Text("ok"))
		body := compattest.Body(t, openrouter.New, &litellm.Request{Model: "m", Messages: []litellm.Message{msg}}, false)
		compattest.AssertJSON(t, body["messages"], `[{"role": "assistant", "content": "ok", "reasoning_details": [{"type": "reasoning.encrypted", "data": "x"}]}]`)
	})
	t.Run("minimax replays text without details", func(t *testing.T) {
		msg := litellm.Assistant(litellm.ReasoningBlock{Text: "t"}, litellm.Text("ok"))
		body := compattest.Body(t, minimax.New, &litellm.Request{Model: "m", Messages: []litellm.Message{msg}}, false)
		compattest.AssertJSON(t, body["messages"], `[{"role": "assistant", "content": "ok", "reasoning_content": "t"}]`)
	})
}

// Every dialect lists its models with the key, OpenRouter from the list that
// takes it, and a rejected key is an auth error.
func TestListModels(t *testing.T) {
	for _, w := range wrappers {
		path := "/v1/models"
		if w.name == "openrouter" {
			path = "/v1/models/user"
		}
		p := compattest.Provider(t, w.newFn, compattest.Doer(func(req *http.Request) (*http.Response, error) {
			if req.Method != http.MethodGet || req.URL.Path != path || req.Header.Get("Authorization") != "Bearer key" {
				t.Errorf("%s: %s %s with %q", w.name, req.Method, req.URL, req.Header.Get("Authorization"))
			}
			return compattest.Response(`{"object":"list","data":[{"id":"a","created":1750000000},{"id":"b","name":"Model B"}]}`), nil
		}))
		got, err := p.ListModels(context.Background())
		want := []litellm.ModelInfo{{ID: "a", Created: time.Unix(1750000000, 0).UTC()}, {ID: "b", Name: "Model B"}}
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("%s: %+v, %v", w.name, got, err)
		}
	}
	p := compattest.Provider(t, compat.New, compattest.Doer(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusUnauthorized, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"bad key"}}`))}, nil
	}))
	if _, err := p.ListModels(context.Background()); litellm.ErrorTypeOf(err) != litellm.ErrorTypeAuth {
		t.Errorf("a rejected key: %v", err)
	}
}
