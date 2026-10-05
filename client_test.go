package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
	"time"
)

type testProvider struct {
	name       string
	chatFunc   func(context.Context, *Request) (*Response, error)
	streamFunc func(context.Context, *Request) (Stream, error)
	lastReq    *Request
}

func (p *testProvider) Name() string { return p.name }

func (p *testProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	p.lastReq = req
	if p.chatFunc != nil {
		return p.chatFunc(ctx, req)
	}
	return &Response{Blocks: []Block{TextBlock{Text: "ok"}}}, nil
}

func (p *testProvider) Stream(ctx context.Context, req *Request) (Stream, error) {
	p.lastReq = req
	if p.streamFunc != nil {
		return p.streamFunc(ctx, req)
	}
	return &testStream{events: append(textEvents(0, "ok"), DoneEvent{FinishReason: FinishReasonStop, Provider: p.name, Model: req.Model})}, nil
}

// testStream replays events, then returns err (io.EOF when nil). Close
// returns closeErr.
type testStream struct {
	events   []Event
	err      error
	closeErr error
	index    int
	closed   bool
}

func (s *testStream) Next() (Event, error) {
	if s.index >= len(s.events) {
		if s.err != nil {
			return nil, s.err
		}
		return nil, io.EOF
	}
	event := s.events[s.index]
	s.index++
	return event, nil
}

func (s *testStream) Close() error {
	s.closed = true
	return s.closeErr
}

type blockingStream struct {
	ctx context.Context
}

func (s blockingStream) Next() (Event, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s blockingStream) Close() error { return nil }

// textEvents streams one complete TextBlock at index.
func textEvents(index int, text string) []Event {
	return []Event{BlockStart{Index: index, Block: TextBlock{}}, TextDelta{Index: index, Text: text}, BlockEnd{Index: index}}
}

var hi = []Message{UserText("hi")}

func TestClientDoesNotInjectDefaults(t *testing.T) {
	provider := &testProvider{name: "test"}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: hi}); err != nil {
		t.Fatal(err)
	}
	if req := provider.lastReq; req.MaxTokens != nil || req.Temperature != nil || req.TopP != nil || req.Thinking != nil {
		t.Fatalf("unexpected defaults injected: %+v", req)
	}
}

func TestClientSendsHistoryUnchanged(t *testing.T) {
	provider := &testProvider{name: "test"}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	// A partial history with a provider-specific ID is the caller's business.
	messages := []Message{
		Assistant(ToolUseBlock{ID: "provider:id!", Name: "tool", Arguments: `{"q":`}),
		UserText("next"),
	}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: messages}); err != nil {
		t.Fatal(err)
	}
	tool := provider.lastReq.Messages[0].Blocks[0].(ToolUseBlock)
	if tool.ID != "provider:id!" || tool.Arguments != `{"q":` || len(provider.lastReq.Messages) != 2 {
		t.Fatalf("Client modified history: %#v", provider.lastReq.Messages)
	}
}

func TestValidateRequest(t *testing.T) {
	inf := math.Inf(1)
	for _, tc := range []struct {
		name string
		req  Request
		want string // empty means accepted
	}{
		{"missing model", Request{Messages: hi}, "model cannot be empty"},
		{"missing messages", Request{Model: "m"}, "messages cannot be empty"},
		{"non-positive max tokens", Request{Model: "m", Messages: hi, MaxTokens: new(0)}, "max_tokens must be positive"},
		{"infinite temperature", Request{Model: "m", Messages: hi, Temperature: &inf}, "temperature must be finite"},
		{"invalid role", Request{Model: "m", Messages: []Message{{Role: "bot", Blocks: []Block{Text("x")}}}}, "invalid role"},
		{"invalid UTF-8 text", Request{Model: "m", Messages: []Message{UserText("hi\xff")}}, "valid UTF-8"},
		{"invalid UTF-8 provider option", Request{Model: "m", Messages: hi, ProviderOptions: ProviderOptions{"k": json.RawMessage{'"', 0xff, '"'}}}, "valid UTF-8"},
		{"reasoning outside assistant", Request{Model: "m", Messages: []Message{User(ReasoningBlock{Text: "x"})}}, "reasoning block requires assistant role"},
		{"tool use without id", Request{Model: "m", Messages: []Message{Assistant(ToolUseBlock{Name: "tool"})}}, "tool use missing id"},
		{"state without provider", Request{Model: "m", Messages: []Message{Assistant(ReasoningBlock{State: &ProviderState{Data: json.RawMessage(`{}`)}})}}, "provider state missing provider"},
		{"state with invalid data", Request{Model: "m", Messages: []Message{Assistant(TextBlock{State: &ProviderState{Provider: "p", Data: json.RawMessage(`{`)}})}}, "provider state data must be valid JSON"},
		{"state with invalid UTF-8", Request{Model: "m", Messages: []Message{Assistant(ToolUseBlock{ID: "c", Name: "t", State: &ProviderState{Provider: "p", Model: "\xff", Data: json.RawMessage(`{}`)}})}}, "provider state must be valid UTF-8"},
		{"tool result outside tool role", Request{Model: "m", Messages: []Message{Assistant(ToolResultBlock{ToolUseID: "call"})}}, "tool result block requires tool role"},
		{"top-level tool reference", Request{Model: "m", Messages: []Message{User(ToolReferenceBlock{ToolName: "lookup"})}}, "only valid inside tool result content"},
		{"tool without name", Request{Model: "m", Messages: hi, Tools: []Tool{{}}}, "tool name cannot be empty"},
		{"invalid tool schema", Request{Model: "m", Messages: hi, Tools: []Tool{{Name: "t", Parameters: Schema(`{`)}}}, "parameters must be valid JSON"},
		{"unknown response format", Request{Model: "m", Messages: hi, ResponseFormat: &ResponseFormat{Type: "xml"}}, "unsupported response format"},
		{"json schema without name", Request{Model: "m", Messages: hi, ResponseFormat: &ResponseFormat{Type: ResponseFormatJSONSchema, JSONSchema: &JSONSchema{}}}, "requires name"},
		{"tool choice mode and name", Request{Model: "m", Messages: hi, ToolChoice: &ToolChoice{Mode: ToolChoiceAuto, Name: "t"}}, "mutually exclusive"},
		{"unknown tool choice mode", Request{Model: "m", Messages: hi, ToolChoice: &ToolChoice{Mode: "any"}}, "unsupported tool choice mode"},
		{"invalid thinking", Request{Model: "m", Messages: hi, Thinking: &Thinking{Disabled: true, Effort: "high"}}, "thinking options cannot be set"},
		// Vendor values are the vendor's to judge.
		{"tool reference inside tool result", Request{Model: "m", Messages: []Message{
			Assistant(ToolUseBlock{ID: "call", Name: "tool"}),
			ToolResult("call", ToolReferenceBlock{ToolName: "lookup"}),
		}}, ""},
		{"vendor values", Request{Model: "any-model", Messages: []Message{User(TextBlock{Text: "x", Cache: &CacheControl{}})},
			Temperature: new(7.5), Thinking: &Thinking{Effort: "ultra", BudgetTokens: new(1 << 30)}, ProviderOptions: ProviderOptions{"anything": json.RawMessage(`1`)}}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client, err := New(&testProvider{name: "test"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.Chat(context.Background(), tc.req)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("rejected: %v", err)
				}
				return
			}
			if ErrorTypeOf(err) != ErrorTypeValidation || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want validation error containing %q", err, tc.want)
			}
		})
	}
}

func TestThinkingValidate(t *testing.T) {
	for _, tc := range []struct {
		name     string
		thinking *Thinking
		want     string
	}{
		{"nil", nil, ""},
		{"zero value is enabled", &Thinking{}, ""},
		{"effort", &Thinking{Effort: "max"}, ""},
		{"budget", &Thinking{BudgetTokens: new(1), IncludeOutput: true}, ""},
		{"disabled", &Thinking{Disabled: true}, ""},
		{"disabled with effort", &Thinking{Disabled: true, Effort: "low"}, "cannot be set when thinking is disabled"},
		{"disabled with output", &Thinking{Disabled: true, IncludeOutput: true}, "cannot be set when thinking is disabled"},
		{"zero budget", &Thinking{BudgetTokens: new(0)}, "must be positive"},
		{"invalid effort", &Thinking{Effort: "\xff"}, "valid UTF-8"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.thinking.validate()
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if ErrorTypeOf(err) != ErrorTypeValidation || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestClientDeepClonesRequestForProvider(t *testing.T) {
	req := Request{
		Model:       "model",
		MaxTokens:   new(10),
		Temperature: new(0.2),
		TopP:        new(0.9),
		Messages: []Message{
			User(TextBlock{Text: "hi", Annotations: []Annotation{{Type: "note", Extra: json.RawMessage(`{"n":1}`)}}}),
			Assistant(ToolUseBlock{ID: "call_1", Name: "tool", Arguments: `{}`, Cache: &CacheControl{}}),
			ToolResultText("call_1", "ok"),
		},
		Tools:          []Tool{{Name: "tool", Parameters: Schema(`{"type":"object"}`)}},
		ToolChoice:     &ToolChoice{Name: "tool"},
		ResponseFormat: &ResponseFormat{Type: ResponseFormatJSONSchema, JSONSchema: &JSONSchema{Name: "out", Schema: Schema(`{"type":"object"}`)}},
		Thinking:       &Thinking{BudgetTokens: new(2048)},
		ProviderOptions: mustProviderOptions(t, map[string]any{
			"metadata": map[string]any{"tags": []any{"a", "b"}},
		}),
	}
	provider := &testProvider{
		name: "test",
		chatFunc: func(ctx context.Context, cloned *Request) (*Response, error) {
			*cloned.MaxTokens, *cloned.Temperature, *cloned.TopP = 99, 1.5, 0.1
			cloned.Tools[0].Parameters[0] = '['
			cloned.ResponseFormat.JSONSchema.Schema[0] = '['
			*cloned.Thinking.BudgetTokens = 4096
			cloned.Messages[0].Blocks[0].(TextBlock).Annotations[0].Extra[0] = '['
			cloned.ToolChoice.Name = "mutated"
			cloned.ProviderOptions["metadata"][0] = '['
			return &Response{Blocks: []Block{Text("ok")}}, nil
		},
	}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if *req.MaxTokens != 10 || *req.Temperature != 0.2 || *req.TopP != 0.9 || *req.Thinking.BudgetTokens != 2048 {
		t.Fatalf("scalar pointers were mutated: %+v", req)
	}
	if string(req.Tools[0].Parameters) != `{"type":"object"}` || string(req.ResponseFormat.JSONSchema.Schema) != `{"type":"object"}` {
		t.Fatal("schemas were mutated")
	}
	if extra := req.Messages[0].Blocks[0].(TextBlock).Annotations[0].Extra; string(extra) != `{"n":1}` {
		t.Fatalf("annotation extra mutated: %s", extra)
	}
	if req.ToolChoice.Name != "tool" || string(req.ProviderOptions["metadata"]) != `{"tags":["a","b"]}` {
		t.Fatal("tool choice or provider options mutated")
	}
}

func TestClientCaptureRawResponse(t *testing.T) {
	raw := []byte(`{"ok":true}`)
	for _, enabled := range []bool{false, true} {
		client, err := New(&testProvider{
			name: "test",
			chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
				return &Response{Blocks: []Block{Text("ok")}, Raw: raw}, nil
			},
		}, WithCaptureRawResponse(enabled))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		want := ""
		if enabled {
			want = string(raw)
		}
		if string(resp.Raw) != want {
			t.Fatalf("enabled=%v raw=%s", enabled, resp.Raw)
		}
	}
}

func TestClientRejectsNilResultsWithoutError(t *testing.T) {
	var endErr error
	client, err := New(&testProvider{
		name:       "test",
		chatFunc:   func(context.Context, *Request) (*Response, error) { return nil, nil },
		streamFunc: func(context.Context, *Request) (Stream, error) { return nil, nil },
	}, WithObservers(endObserver(func(r CallResult) { endErr = r.Err })))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = client.Chat(context.Background(), Request{Model: "m", Messages: hi}); err == nil || !strings.Contains(err.Error(), "nil response without error") {
		t.Fatalf("chat error = %v", err)
	}
	if _, err = client.Stream(context.Background(), Request{Model: "m", Messages: hi}); err == nil || !strings.Contains(err.Error(), "nil stream without error") {
		t.Fatalf("stream error = %v", err)
	}
	if endErr == nil || !strings.Contains(endErr.Error(), "nil stream without error") {
		t.Fatalf("End received %v, want nil stream error", endErr)
	}
}

func TestClientWarnsOnMalformedToolArguments(t *testing.T) {
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(context.Context, *Request) (*Response, error) {
			return &Response{FinishReason: FinishReasonToolCall, Blocks: []Block{ToolUseBlock{ID: "call_bad", Name: "lookup", Arguments: `{"q":`}}}, nil
		},
		streamFunc: func(context.Context, *Request) (Stream, error) {
			return &testStream{events: []Event{
				BlockStart{Index: 0, Block: ToolUseBlock{ID: "call_bad", Name: "lookup"}},
				ToolUseDelta{Index: 0, Arguments: `{"q":`},
				BlockEnd{Index: 0, Block: ToolUseBlock{}},
				DoneEvent{FinishReason: FinishReasonToolCall},
			}}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: hi})
	if err != nil {
		t.Fatal(err)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].Arguments != `{"q":` {
		t.Fatalf("tool calls = %#v, want raw malformed args", calls)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "litellm.tool_arguments_invalid" || resp.Warnings[0].Provider != "test" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
	if resp.Provider != "test" || resp.Model != "m" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: hi})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if resp, err = Collect(stream); err != nil {
		t.Fatal(err)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "litellm.tool_arguments_invalid" {
		t.Fatalf("stream warnings = %+v, want exactly one", resp.Warnings)
	}
}

// A tool call without arguments has the empty object, whether the vendor
// omitted them or streamed no deltas.
func TestClientCompletesArgumentlessToolCalls(t *testing.T) {
	call := ToolUseBlock{ID: "call", Name: "noop"}
	client, err := New(&testProvider{
		name:     "test",
		chatFunc: func(context.Context, *Request) (*Response, error) { return &Response{Blocks: []Block{call}}, nil },
		streamFunc: func(context.Context, *Request) (Stream, error) {
			return &testStream{events: []Event{BlockStart{Index: 0, Block: call}, BlockEnd{Index: 0}, DoneEvent{}}}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: hi})
	if err != nil || resp.ToolCalls()[0].Arguments != "{}" || len(resp.Warnings) != 0 {
		t.Fatalf("chat = %+v, %v", resp, err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: hi})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if resp, err = Collect(stream); err != nil || resp.ToolCalls()[0].Arguments != "{}" || len(resp.Warnings) != 0 {
		t.Fatalf("stream = %+v, %v", resp, err)
	}
}

func TestClientWrapsProviderErrors(t *testing.T) {
	boom := errors.New("boom")
	provider := &testProvider{
		name:     "test",
		chatFunc: func(context.Context, *Request) (*Response, error) { return nil, boom },
		streamFunc: func(_ context.Context, req *Request) (Stream, error) {
			if req.Model == "start" {
				return nil, boom
			}
			return &testStream{events: textEvents(0, "partial")[:2], err: boom}, nil
		},
	}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	_, chatErr := client.Chat(ctx, Request{Model: "m", Messages: hi})
	_, startErr := client.Stream(ctx, Request{Model: "start", Messages: hi})
	stream, err := client.Stream(ctx, Request{Model: "m", Messages: hi})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	resp, runtimeErr := Collect(stream)
	for name, err := range map[string]error{"chat": chatErr, "stream start": startErr, "stream runtime": runtimeErr} {
		if ErrorTypeOf(err) != ErrorTypeProvider || !errors.Is(err, boom) || !strings.HasPrefix(err.Error(), "test: ") {
			t.Errorf("%s: err = %v, want wrapped provider error", name, err)
		}
	}
	if resp.Text() != "partial" {
		t.Fatalf("partial response = %#v", resp)
	}
}

func mustProviderOptions(t *testing.T, values map[string]any) ProviderOptions {
	t.Helper()
	o, err := NewProviderOptions(values)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func TestProviderOptionsJSONBoundary(t *testing.T) {
	source := map[string]any{"tags": []string{"original"}, "limit": int64(9007199254740993)}
	options, err := NewProviderOptions(map[string]any{"config": source})
	if err != nil {
		t.Fatal(err)
	}
	source["tags"].([]string)[0] = "changed"
	decoded, err := options.Decode()
	if err != nil {
		t.Fatal(err)
	}
	config := decoded["config"].(map[string]any)
	if config["tags"].([]any)[0] != "original" || config["limit"].(json.Number).String() != "9007199254740993" {
		t.Fatalf("decoded = %#v", config)
	}
	config["tags"].([]any)[0] = "decoder changed"
	second, err := options.Decode()
	if err != nil {
		t.Fatal(err)
	}
	if second["config"].(map[string]any)["tags"].([]any)[0] != "original" {
		t.Fatal("Decode shares storage")
	}
	var unsupported *json.UnsupportedTypeError
	if err := options.Set("invalid", func() {}); ErrorTypeOf(err) != ErrorTypeValidation || !errors.As(err, &unsupported) {
		t.Fatalf("encoding error = %v", err)
	}
	if _, exists := options["invalid"]; exists {
		t.Fatal("failed Set changed options")
	}
	for _, raw := range []json.RawMessage{nil, json.RawMessage(`{"unterminated":`), json.RawMessage(`1 2`), {'"', 0xff, '"'}} {
		if err := (ProviderOptions{"invalid": raw}).validate(); ErrorTypeOf(err) != ErrorTypeValidation {
			t.Fatalf("raw %q: %v", raw, err)
		}
	}
}

func TestLaterUsageReplacesEarlier(t *testing.T) {
	collector := newCollector()
	for _, usage := range []Usage{{InputTokens: 5}, {OutputTokens: 2}} {
		if _, _, err := collector.Apply(UsageEvent{Usage: usage}); err != nil {
			t.Fatal(err)
		}
	}
	if got := collector.Response().Usage; got != (Usage{OutputTokens: 2}) {
		t.Fatalf("usage = %+v, want the last snapshot", got)
	}
}

// causeStream fails as net/http does once its context ends: with the
// context's cause, which need not be context.Canceled.
type causeStream struct{ ctx context.Context }

func (s causeStream) Next() (Event, error) {
	<-s.ctx.Done()
	return nil, NewNetworkError("test", "stream read error", context.Cause(s.ctx))
}

func (s causeStream) Close() error { return nil }

// A call the caller's context ends is canceled or timed out, and not
// temporary, whatever error the request failed with.
func TestCallEndedByTheCallersContext(t *testing.T) {
	provider := &testProvider{
		name: "test",
		chatFunc: func(ctx context.Context, _ *Request) (*Response, error) {
			_, err := causeStream{ctx}.Next()
			return nil, err
		},
		streamFunc: func(ctx context.Context, _ *Request) (Stream, error) { return causeStream{ctx}, nil },
	}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	stop := errors.New("user stopped")
	for _, tc := range []struct {
		name string
		ctx  func() (context.Context, context.CancelFunc)
		want ErrorType
		is   error
	}{
		{"canceled with a cause", func() (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancelCause(context.Background())
			cancel(stop)
			return ctx, func() {}
		}, ErrorTypeCanceled, context.Canceled},
		{"timed out with a cause", func() (context.Context, context.CancelFunc) {
			return context.WithTimeoutCause(context.Background(), time.Millisecond, stop)
		}, ErrorTypeTimeout, context.DeadlineExceeded},
	} {
		t.Run(tc.name, func(t *testing.T) {
			check := func(err error) {
				t.Helper()
				if ErrorTypeOf(err) != tc.want || IsTemporaryError(err) || !errors.Is(err, tc.is) || !errors.Is(err, stop) {
					t.Fatalf("err = %v (type %q, temporary %v)", err, ErrorTypeOf(err), IsTemporaryError(err))
				}
			}
			ctx, cancel := tc.ctx()
			defer cancel()
			_, err := client.Chat(ctx, Request{Model: "m", Messages: hi})
			check(err)
			stream, err := client.Stream(ctx, Request{Model: "m", Messages: hi})
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			_, err = Collect(stream)
			check(err)
		})
	}
}

// A reply that stops with tool calls ended for them, whichever way it came;
// one cut short by its length did not.
func TestStopWithToolCallsIsToolCall(t *testing.T) {
	call := ToolUseBlock{ID: "c", Name: "f", Arguments: "{}"}
	for _, tc := range []struct {
		finish, want FinishReason
	}{
		{FinishReasonStop, FinishReasonToolCall},
		{FinishReasonLength, FinishReasonLength},
	} {
		p := &testProvider{
			name: "test",
			chatFunc: func(context.Context, *Request) (*Response, error) {
				return &Response{Blocks: []Block{call}, FinishReason: tc.finish, FinishReasonRaw: "raw"}, nil
			},
			streamFunc: func(_ context.Context, req *Request) (Stream, error) {
				return &testStream{events: []Event{
					BlockStart{Index: 0, Block: ToolUseBlock{ID: "c", Name: "f"}}, ToolUseDelta{Index: 0, Arguments: "{}"}, BlockEnd{Index: 0},
					DoneEvent{FinishReason: tc.finish, FinishReasonRaw: "raw", Provider: "test", Model: req.Model},
				}}, nil
			},
		}
		client, _ := New(p)
		req := Request{Model: "m", Messages: []Message{UserText("hi")}}
		resp, err := client.Chat(t.Context(), req)
		if err != nil || resp.FinishReason != tc.want || resp.FinishReasonRaw != "raw" {
			t.Fatalf("%s: chat = %+v, %v", tc.finish, resp, err)
		}
		stream, err := client.Stream(t.Context(), req)
		if err != nil {
			t.Fatal(err)
		}
		var done DoneEvent
		resp, err = Handle(stream, func(e Event) error {
			if d, ok := e.(DoneEvent); ok {
				done = d
			}
			return nil
		})
		stream.Close()
		if err != nil || resp.FinishReason != tc.want || done.FinishReason != tc.want || done.FinishReasonRaw != "raw" {
			t.Fatalf("%s: stream = %+v, done = %+v, %v", tc.finish, resp, done, err)
		}
	}
}

func TestNewRejectsUnnamedProvider(t *testing.T) {
	if _, err := New(&testProvider{}); err == nil {
		t.Fatal("accepted a provider without a name")
	}
}
