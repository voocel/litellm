package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"reflect"
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
	return &testStream{events: []Event{ContentDelta{Text: "ok"}, DoneEvent{FinishReason: FinishReasonStop, Provider: p.name, Model: req.Model}}}, nil
}

type testModelProvider struct {
	*testProvider
	listFunc func(context.Context) ([]ModelInfo, error)
}

func (p *testModelProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	if p.listFunc != nil {
		return p.listFunc(ctx)
	}
	return []ModelInfo{{ID: "m", Provider: p.Name()}}, nil
}

type testStream struct {
	events []Event
	index  int
	closed bool
}

func (s *testStream) Next() (Event, error) {
	if s.index >= len(s.events) {
		return nil, io.EOF
	}
	event := s.events[s.index]
	s.index++
	return event, nil
}

func (s *testStream) Close() error {
	s.closed = true
	return nil
}

type blockingStream struct {
	ctx context.Context
}

func (s blockingStream) Next() (Event, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s blockingStream) Close() error {
	return nil
}

func TestClientDoesNotInjectDefaults(t *testing.T) {
	provider := &testProvider{name: "test"}
	client, err := New(provider)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if provider.lastReq.MaxTokens != nil || provider.lastReq.Temperature != nil || provider.lastReq.TopP != nil {
		t.Fatalf("unexpected defaults injected: %+v", provider.lastReq)
	}
}

func TestJSONRawReturnsMarshalError(t *testing.T) {
	_, err := JSONRaw(math.Inf(1))
	if err == nil {
		t.Fatalf("expected marshal error")
	}
}

func TestThinkingOptionsRequireExplicitMode(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
		Thinking: &Thinking{Effort: "high"},
	})
	if err == nil || !strings.Contains(err.Error(), "thinking mode must be enabled or disabled") {
		t.Fatalf("expected thinking mode error, got %v", err)
	}
}

func TestThinkingDisabledRejectsOptions(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
		Thinking: &Thinking{Mode: ThinkingDisabled, Effort: "high"},
	})
	if err == nil || !strings.Contains(err.Error(), "thinking options cannot be set when thinking is disabled") {
		t.Fatalf("expected disabled thinking options error, got %v", err)
	}
}

func TestClientAppliesExplicitDefaults(t *testing.T) {
	maxTokens := 123
	temp := 0.4
	client, err := New(&testProvider{name: "test"}, WithDefaults(RequestDefaults{
		MaxTokens:   &maxTokens,
		Temperature: &temp,
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "ok" {
		t.Fatalf("response text = %q", resp.Text())
	}
}

func TestClientDeepClonesRequestForProvider(t *testing.T) {
	maxTokens := 10
	temp := 0.2
	topP := 0.9
	budget := 2048
	schema := Schema(`{"type":"object"}`)
	req := Request{
		Model:       "model",
		MaxTokens:   &maxTokens,
		Temperature: &temp,
		TopP:        &topP,
		Messages: []Message{
			User(TextBlock{
				Text: "hi",
				Annotations: []Annotation{
					{Type: "note", Extra: MustJSONRaw(map[string]any{"n": 1})},
				},
			}),
			Assistant(ToolUseBlock{
				ID:        "call_1",
				Name:      "tool",
				Arguments: MustJSONRaw(map[string]any{}),
				Cache:     &CacheControl{Type: CacheTypeEphemeral, TTL: CacheTTL1h},
			}),
			ToolResultText("call_1", "ok"),
		},
		ToolChoice: &ToolChoice{Name: "tool"},
		ResponseFormat: &ResponseFormat{
			Type: ResponseFormatJSONSchema,
			JSONSchema: &JSONSchema{
				Name:   "out",
				Schema: schema,
			},
		},
		Thinking: &Thinking{Mode: ThinkingEnabled, BudgetTokens: &budget},
		Cache:    &CachePolicy{Retention: CacheTTL1h, Placement: CachePlacementPrefix},
		ProviderOptions: mustProviderOptions(t, map[string]any{
			"metadata": map[string]any{
				"tags":   []any{"a", "b"},
				"nested": map[string]any{"k": "v"},
			},
		}),
	}
	provider := &testProvider{
		name: "test",
		chatFunc: func(ctx context.Context, cloned *Request) (*Response, error) {
			*cloned.MaxTokens = 99
			*cloned.Temperature = 1.5
			*cloned.TopP = 0.1
			cloned.ResponseFormat.JSONSchema.Schema[0] = '['
			*cloned.Thinking.BudgetTokens = 4096
			cloned.Cache.Retention = CacheTTL5m
			text := cloned.Messages[0].Blocks[0].(TextBlock)
			text.Annotations[0].Extra[0] = '['
			cloned.Messages[0].Blocks[0] = text
			tool := cloned.Messages[1].Blocks[0].(ToolUseBlock)
			tool.Arguments[0] = '['
			tool.Cache.TTL = CacheTTL5m
			cloned.Messages[1].Blocks[0] = tool
			cloned.ToolChoice.Name = "mutated"
			cloned.ProviderOptions["metadata"][0] = '['

			return &Response{Blocks: []Block{Text("ok")}}, nil
		},
	}
	client, err := New(provider)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if _, err := client.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if *req.MaxTokens != 10 || *req.Temperature != 0.2 || *req.TopP != 0.9 {
		t.Fatalf("scalar pointers were mutated: %+v", req)
	}
	if string(req.ResponseFormat.JSONSchema.Schema) != `{"type":"object"}` {
		t.Fatalf("schema was mutated: %s", req.ResponseFormat.JSONSchema.Schema)
	}
	if *req.Thinking.BudgetTokens != 2048 || req.Cache.Retention != CacheTTL1h {
		t.Fatalf("thinking/cache mutated: %+v %+v", req.Thinking, req.Cache)
	}
	text := req.Messages[0].Blocks[0].(TextBlock)
	if string(text.Annotations[0].Extra) != `{"n":1}` {
		t.Fatalf("annotation extra mutated: %s", text.Annotations[0].Extra)
	}
	tool := req.Messages[1].Blocks[0].(ToolUseBlock)
	if string(tool.Arguments) != `{}` || tool.Cache.TTL != CacheTTL1h {
		t.Fatalf("tool block mutated: %#v", tool)
	}
	if req.ToolChoice.Name != "tool" {
		t.Fatalf("tool choice mutated: %#v", req.ToolChoice)
	}
	if string(req.ProviderOptions["metadata"]) != `{"nested":{"k":"v"},"tags":["a","b"]}` {
		t.Fatal("provider options mutated")
	}
}

func TestValidateRejectsInvalidScalars(t *testing.T) {
	temp := math.Inf(1)
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:       "model",
		Messages:    []Message{UserText("hi")},
		Temperature: &temp,
	})
	if err == nil || !IsValidationError(err) {
		t.Fatalf("expected validation error, got %v", err)
	}
}

func TestValidateRejectsInvalidCachePolicy(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
		Cache:    &CachePolicy{Retention: "forever", Placement: CachePlacementPrefix},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "cache retention") {
		t.Fatalf("expected cache retention validation error, got %v", err)
	}

	_, err = client.Chat(context.Background(), Request{
		Model:    "model",
		Messages: []Message{UserText("hi")},
		Cache:    &CachePolicy{Retention: CacheTTL1h, Placement: CachePlacement("suffix")},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "cache placement") {
		t.Fatalf("expected cache placement validation error, got %v", err)
	}
}

func TestValidateRejectsInvalidBlockCacheControl(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model: "model",
		Messages: []Message{
			User(TextBlock{Text: "hi", Cache: &CacheControl{Type: "persistent"}}),
		},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "cache type") {
		t.Fatalf("expected cache type validation error, got %v", err)
	}

	_, err = client.Chat(context.Background(), Request{
		Model: "model",
		Messages: []Message{
			User(TextBlock{Text: "hi", Cache: &CacheControl{Type: CacheTypeEphemeral, TTL: "24h"}}),
		},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "cache ttl") {
		t.Fatalf("expected cache ttl validation error, got %v", err)
	}
}

func TestValidateRejectsInvalidUTF8Text(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model: "model",
		Messages: []Message{
			UserText(string([]byte{'h', 'i', 0xff})),
		},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "valid UTF-8") {
		t.Fatalf("expected UTF-8 validation error, got %v", err)
	}
}

func TestValidateRejectsInvalidUTF8ProviderOptions(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:           "model",
		Messages:        []Message{UserText("hi")},
		ProviderOptions: ProviderOptions{"metadata": json.RawMessage{'"', 0xff, '"'}},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "valid UTF-8") {
		t.Fatalf("expected UTF-8 validation error, got %v", err)
	}
}

func TestStreamIdleTimeout(t *testing.T) {
	client, err := New(&testProvider{
		name: "test",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return blockingStream{ctx: ctx}, nil
		},
	}, WithStreamIdleTimeout(10*time.Millisecond))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	defer stream.Close()

	_, err = stream.Next()
	if err == nil || !IsTimeoutError(err) || !IsStreamIdleError(err) {
		t.Fatalf("expected stream idle timeout, got %v", err)
	}
}

func TestStreamIdleTimeoutStopsAfterDoneEvent(t *testing.T) {
	client, err := New(&testProvider{
		name: "test",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &testStream{events: []Event{DoneEvent{FinishReason: FinishReasonStop}}}, nil
		},
	}, WithStreamIdleTimeout(10*time.Millisecond))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	defer stream.Close()

	event, err := stream.Next()
	if err != nil {
		t.Fatalf("Next returned error: %v", err)
	}
	if _, ok := event.(DoneEvent); !ok {
		t.Fatalf("event = %#v, want DoneEvent", event)
	}
	time.Sleep(20 * time.Millisecond)
	_, err = stream.Next()
	if !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF after done, got %v", err)
	}
}

func TestClientObservers(t *testing.T) {
	var before, after int
	client, err := New(&testProvider{name: "hook"}, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
		meta := info

		before++
		if meta.Provider != "hook" || meta.Model != "m" {
			t.Fatalf("bad meta: %+v", meta)
		}

		return ctx, CallObserverFuncs{EndFunc: func(result CallResult) {
			resp := result.Response
			err := result.Err

			after++
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if resp == nil || resp.Text() != "ok" {
				t.Fatalf("bad hook response: %#v", resp)
			}

		}}
	})))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if before != 1 || after != 1 || resp.Text() != "ok" {
		t.Fatalf("hooks before=%d after=%d resp=%v", before, after, resp)
	}
}

func TestObserversCannotMutateProviderRequestOrReturnedResponse(t *testing.T) {
	provider := &testProvider{
		name: "hook",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			if req.Model != "m" {
				t.Fatalf("provider saw mutated model %q", req.Model)
			}
			if got := req.Messages[0].Blocks[0].(TextBlock).Text; got != "hi" {
				t.Fatalf("provider saw mutated message %q", got)
			}
			return &Response{Blocks: []Block{Text("ok")}, Warnings: []Warning{{Code: "w"}}}, nil
		},
	}
	client, err := New(provider, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
		req := info.Request

		req.Model = "mutated"
		req.Messages[0].Blocks[0] = Text("mutated")

		return ctx, CallObserverFuncs{EndFunc: func(result CallResult) {
			resp := result.Response

			resp.Blocks[0] = Text("mutated")
			resp.Warnings[0].Code = "mutated"

		}}
	})))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "ok" {
		t.Fatalf("response text = %q, want ok", resp.Text())
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "w" {
		t.Fatalf("warnings = %#v", resp.Warnings)
	}
}

func TestObserversCannotMutateStructProviderOptions(t *testing.T) {
	type options struct {
		Tags     []string
		Metadata map[string]string
		Limit    *int
		Created  time.Time
	}
	newOptions := func() options {
		return options{
			Tags:     []string{"original"},
			Metadata: map[string]string{"key": "original"},
			Limit:    IntPtr(10),
			Created:  time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		}
	}
	original, want := newOptions(), newOptions()
	checkOptions := func(got any) {
		t.Helper()
		if !reflect.DeepEqual(got, want) {
			t.Errorf("options = %#v, want %#v", got, want)
		}
	}
	provider := &testProvider{
		name: "hook",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			var decoded options
			if err := json.Unmarshal(req.ProviderOptions["custom"], &decoded); err != nil {
				t.Fatal(err)
			}
			checkOptions(decoded)
			req.ProviderOptions["custom"][0] = '['
			return &Response{Blocks: []Block{Text("ok")}}, nil
		},
	}
	client, err := New(provider, WithObservers(
		ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
			req := info.Request

			req.ProviderOptions["custom"][0] = '['

			return ctx, CallObserverFuncs{}
		}),
		ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
			req := info.Request

			var decoded options
			if err := json.Unmarshal(req.ProviderOptions["custom"], &decoded); err != nil {
				t.Fatal(err)
			}
			checkOptions(decoded)

			return ctx, CallObserverFuncs{}
		}),
	))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:           "m",
		Messages:        []Message{UserText("hi")},
		ProviderOptions: mustProviderOptions(t, map[string]any{"custom": original}),
	})
	if err != nil {
		t.Fatal(err)
	}
	checkOptions(original)
}

func TestClientCaptureRawResponseOption(t *testing.T) {
	raw := []byte(`{"ok":true}`)
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			resp := &Response{Blocks: []Block{Text("ok")}}
			CaptureRawResponse(req, resp, raw)
			return resp, nil
		},
	}, WithCaptureRawResponse(true))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if string(resp.Raw) != string(raw) {
		t.Fatalf("raw = %s, want %s", resp.Raw, raw)
	}
}

func TestClientDoesNotCaptureRawResponseByDefault(t *testing.T) {
	raw := []byte(`{"ok":true}`)
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			resp := &Response{Blocks: []Block{Text("ok")}}
			CaptureRawResponse(req, resp, raw)
			return resp, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if len(resp.Raw) != 0 {
		t.Fatalf("raw should be empty by default: %s", resp.Raw)
	}
}

func TestClientRejectsNilResponseWithoutError(t *testing.T) {
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(context.Context, *Request) (*Response, error) {
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err == nil || !strings.Contains(err.Error(), "nil response without error") {
		t.Fatalf("error = %v, want nil response contract", err)
	}
}

func TestClientAllowsMalformedToolArgumentsFromModel(t *testing.T) {
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(context.Context, *Request) (*Response, error) {
			return &Response{
				Provider:     "test",
				Model:        "m",
				FinishReason: FinishReasonToolCall,
				Blocks: []Block{
					ToolUseBlock{ID: "call_bad", Name: "lookup", Arguments: []byte(`{"q":`)},
				},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("tool calls len = %d, want 1", len(calls))
	}
	if got := string(calls[0].Arguments); got != `{"q":` {
		t.Fatalf("arguments = %q, want raw malformed args", got)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "tool_arguments_invalid" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

func TestClientRejectsNilStreamWithoutError(t *testing.T) {
	var hookErr error
	client, err := New(&testProvider{
		name: "test",
		streamFunc: func(context.Context, *Request) (Stream, error) {
			return nil, nil
		},
	}, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {

		return ctx, CallObserverFuncs{EndFunc: func(result CallResult) {
			err := result.Err

			hookErr = err

		}}
	})))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err == nil || !strings.Contains(err.Error(), "nil stream without error") {
		t.Fatalf("expected nil stream error, got %v", err)
	}
	if hookErr == nil || !strings.Contains(hookErr.Error(), "nil stream without error") {
		t.Fatalf("End received %v, want nil stream error", hookErr)
	}
}

func TestClientWrapsProviderChatErrors(t *testing.T) {
	boom := errors.New("boom")
	client, err := New(&testProvider{
		name: "test",
		chatFunc: func(context.Context, *Request) (*Response, error) {
			return nil, boom
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err == nil || !IsProviderError(err) || !errors.Is(err, boom) {
		t.Fatalf("expected wrapped provider error, got %v", err)
	}
}

func TestClientWrapsProviderStreamStartErrors(t *testing.T) {
	boom := errors.New("boom")
	client, err := New(&testProvider{
		name: "test",
		streamFunc: func(context.Context, *Request) (Stream, error) {
			return nil, boom
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err == nil || !IsProviderError(err) || !errors.Is(err, boom) {
		t.Fatalf("expected wrapped provider stream error, got %v", err)
	}
}

func TestWrapErrorClassifiesContextErrors(t *testing.T) {
	canceled := WrapError(context.Canceled, "test")
	if !IsNetworkError(canceled) || IsTemporaryError(canceled) || !errors.Is(canceled, context.Canceled) {
		t.Fatalf("canceled error = %v", canceled)
	}
	deadline := WrapError(context.DeadlineExceeded, "test")
	if !IsTimeoutError(deadline) || IsTemporaryError(deadline) || !errors.Is(deadline, context.DeadlineExceeded) {
		t.Fatalf("deadline error = %v", deadline)
	}
}

func TestClientListModelsWrapsProviderErrors(t *testing.T) {
	boom := errors.New("boom")
	client, err := New(&testModelProvider{
		testProvider: &testProvider{name: "test"},
		listFunc: func(context.Context) ([]ModelInfo, error) {
			return nil, boom
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.ListModels(context.Background())
	if err == nil || !IsProviderError(err) || !errors.Is(err, boom) {
		t.Fatalf("expected wrapped model list error, got %v", err)
	}
}

func TestCollect(t *testing.T) {
	stream := &testStream{events: []Event{
		ReasoningDelta{Text: "think"},
		ContentDelta{Text: "hel"},
		ContentDelta{Text: "lo"},
		ToolUseStart{ID: "call_1", Name: "tool"},
		ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`{"x":`)},
		ToolUseDelta{ID: "call_1", ArgumentsDelta: []byte(`1}`)},
		DoneEvent{FinishReason: FinishReasonToolCall, Provider: "test", Model: "m"},
	}}
	resp, err := Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "hello" {
		t.Fatalf("text = %q", resp.Text())
	}
	if resp.Reasoning() != "think" {
		t.Fatalf("reasoning = %q", resp.Reasoning())
	}
	if resp.Provider != "test" || resp.Model != "m" {
		t.Fatalf("provider/model = %q/%q", resp.Provider, resp.Model)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || string(calls[0].Arguments) != `{"x":1}` {
		t.Fatalf("tool calls = %+v", calls)
	}
}

func TestHistoryValidationIsExplicit(t *testing.T) {
	provider := &testProvider{name: "test"}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	messages := []Message{
		Assistant(ToolUseBlock{ID: "provider:id!", Name: "tool", Arguments: MustJSONRaw(map[string]any{})}),
		UserText("next"),
	}
	if err := ValidateHistory(messages); !IsValidationError(err) {
		t.Fatalf("expected explicit history validation error, got %v", err)
	}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: messages}); err != nil {
		t.Fatalf("Client must allow provider-specific IDs and partial histories: %v", err)
	}
	if got := provider.lastReq.Messages[0].Blocks[0].(ToolUseBlock).ID; got != "provider:id!" {
		t.Fatalf("Client modified history: %q", got)
	}
}

func TestValidateRejectsToolResultOutsideToolRole(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			Assistant(
				ToolUseBlock{ID: "call_1", Name: "tool", Arguments: MustJSONRaw(map[string]any{})},
				ToolResultBlock{ToolUseID: "call_1", Content: []Block{Text("ok")}},
			),
		},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "tool result block requires tool role") {
		t.Fatalf("expected tool role validation error, got %v", err)
	}
}

func TestValidateRejectsTopLevelToolReference(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model:    "m",
		Messages: []Message{User(ToolReferenceBlock{ToolName: "lookup"})},
	})
	if err == nil || !IsValidationError(err) || !strings.Contains(err.Error(), "tool reference block is only valid inside tool result content") {
		t.Fatalf("expected tool reference validation error, got %v", err)
	}
}

func TestValidateAllowsToolReferenceInsideToolResult(t *testing.T) {
	client, err := New(&testProvider{name: "test"})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.Chat(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			Assistant(ToolUseBlock{ID: "call_1", Name: "tool", Arguments: MustJSONRaw(map[string]any{})}),
			ToolResult("call_1", ToolReferenceBlock{ToolName: "lookup"}),
		},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
}

func TestMessageRepairIsExplicitAndWarns(t *testing.T) {
	original := []Message{
		Assistant(ToolUseBlock{ID: "bad id!", Name: "tool", Arguments: MustJSONRaw(map[string]any{})}),
		UserText("next"),
	}
	messages, warnings := RepairMessages(original, RepairAll)
	if len(warnings) != 2 {
		t.Fatalf("warnings = %#v", warnings)
	}
	if err := ValidateHistory(messages); err != nil {
		t.Fatal(err)
	}
	if got := messages[0].Blocks[0].(ToolUseBlock).ID; got != "bad_id_" {
		t.Fatalf("tool use id = %q", got)
	}
	if got := original[0].Blocks[0].(ToolUseBlock).ID; got != "bad id!" {
		t.Fatalf("input mutated: %q", got)
	}
	if messages[1].Role != RoleTool || messages[2].Role != RoleUser {
		t.Fatalf("messages = %#v", messages)
	}
	provider := &testProvider{name: "test"}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Warnings) != 0 {
		t.Fatalf("repair warnings must remain with caller: %#v", resp.Warnings)
	}
}

func TestMessageRepairSynthesizesMissingToolUseID(t *testing.T) {
	messages, warnings := RepairMessages([]Message{
		Assistant(ToolUseBlock{Name: "tool", Arguments: MustJSONRaw(map[string]any{})}),
	}, RepairAll)
	if len(warnings) != 2 {
		t.Fatalf("warnings = %#v", warnings)
	}
	if err := ValidateHistory(messages); err != nil {
		t.Fatal(err)
	}
	generated := messages[0].Blocks[0].(ToolUseBlock).ID
	if !strings.HasPrefix(generated, "call_") {
		t.Fatalf("generated id = %q", generated)
	}
	result := messages[1].Blocks[0].(ToolResultBlock)
	if result.ToolUseID != generated || !result.IsError {
		t.Fatalf("synthetic result = %#v, generated=%q", result, generated)
	}
}

func TestMessageRepairInsertsMissingToolResultsInToolUseOrder(t *testing.T) {
	messages, warnings := RepairMessages([]Message{
		Assistant(
			ToolUseBlock{ID: "call_1", Name: "first", Arguments: MustJSONRaw(map[string]any{})},
			ToolUseBlock{ID: "call_2", Name: "second", Arguments: MustJSONRaw(map[string]any{})},
		),
		UserText("next"),
	}, RepairInsertMissingToolResults)
	if len(warnings) != 2 {
		t.Fatalf("warnings len = %d, want 2: %#v", len(warnings), warnings)
	}
	if len(messages) < 3 {
		t.Fatalf("messages = %#v", messages)
	}
	first, ok := messages[1].Blocks[0].(ToolResultBlock)
	if !ok || first.ToolUseID != "call_1" {
		t.Fatalf("first synthetic result = %#v", messages[1].Blocks[0])
	}
	second, ok := messages[2].Blocks[0].(ToolResultBlock)
	if !ok || second.ToolUseID != "call_2" {
		t.Fatalf("second synthetic result = %#v", messages[2].Blocks[0])
	}
}

func TestStreamObserverEndsOnError(t *testing.T) {
	boom := errors.New("boom")
	var endErr error
	client, err := New(&testProvider{
		name: "stream",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &testStreamWithError{events: []Event{ContentDelta{Text: "partial"}}, err: boom}, nil
		},
	}, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {

		return ctx, CallObserverFuncs{EndFunc: func(result CallResult) {
			err := result.Err

			endErr = err

		}}
	})))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	_, err = Collect(stream)
	if !errors.Is(err, boom) {
		t.Fatalf("Collect err = %v, want %v", err, boom)
	}
	if !errors.Is(endErr, boom) {
		t.Fatalf("end err = %v, want %v", endErr, boom)
	}
}

func TestStreamWrapsRuntimeProviderErrors(t *testing.T) {
	boom := errors.New("boom")
	client, err := New(&testProvider{
		name: "stream",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &testStreamWithError{events: []Event{ContentDelta{Text: "partial"}}, err: boom}, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	_, err = Collect(stream)
	if err == nil || !IsProviderError(err) || !errors.Is(err, boom) {
		t.Fatalf("expected wrapped runtime stream error, got %v", err)
	}
}

func TestStreamObserversCannotMutateReturnedEvents(t *testing.T) {
	newEvents := func() []Event {
		return []Event{
			ContentStart{Block: ReasoningBlock{Extra: []byte(`{}`)}, OutputIndex: IntPtr(2), ContentIndex: IntPtr(0)},
			ContentEnd{OutputIndex: IntPtr(2), ContentIndex: IntPtr(0)},
			ContentStart{Block: TextBlock{}, OutputIndex: IntPtr(3), ContentIndex: IntPtr(0)},
			ContentEnd{Block: TextBlock{Annotations: []Annotation{{Extra: []byte(`{}`)}}}, OutputIndex: IntPtr(3), ContentIndex: IntPtr(0)},
			ContentDelta{Text: "hi", OutputIndex: IntPtr(0), ContentIndex: IntPtr(1)},
			RefusalDelta{Text: "no", OutputIndex: IntPtr(0), ContentIndex: IntPtr(1)},
			ReasoningDelta{Text: "thinking", ContentIndex: IntPtr(0), Redacted: []byte("data"), Extra: []byte(`{}`)},
			ToolUseStart{ID: "call_1", Name: "lookup", Index: IntPtr(0), OutputIndex: IntPtr(1)},
			ToolUseDelta{ID: "call_1", Index: IntPtr(0), OutputIndex: IntPtr(1), ArgumentsDelta: []byte(`{"q":"x"}`)},
			ToolUseDone{ID: "call_1", Index: IntPtr(0), OutputIndex: IntPtr(1)},
			ProviderEvent{Name: "provider.event", Raw: []byte(`{"ok":true}`)},
			ContentDelta{Text: "without indices"},
			DoneEvent{FinishReason: FinishReasonStop, Provider: "stream", Model: "m"},
		}
	}
	events, want := newEvents(), newEvents()
	observed := 0
	client, err := New(&testProvider{
		name: "stream",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &testStream{events: events}, nil
		},
	}, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {

		return ctx, CallObserverFuncs{OnEventFunc: func(event Event) {

			switch e := event.(type) {
			case ContentStart:
				*e.OutputIndex, *e.ContentIndex = 99, 99
				if block, ok := e.Block.(ReasoningBlock); ok {
					block.Extra[0] = '['
				}
			case ContentEnd:
				*e.OutputIndex, *e.ContentIndex = 99, 99
				if block, ok := e.Block.(TextBlock); ok {
					block.Annotations[0].Extra[0] = '['
				}
			case ContentDelta:
				if e.OutputIndex != nil {
					*e.OutputIndex, *e.ContentIndex = 99, 99
				}
			case RefusalDelta:
				*e.OutputIndex, *e.ContentIndex = 99, 99
			case ReasoningDelta:
				*e.ContentIndex = 99
				e.Redacted[0], e.Extra[0] = 'x', '['
			case ToolUseStart:
				*e.Index, *e.OutputIndex = 99, 99
			case ToolUseDelta:
				*e.Index, *e.OutputIndex = 99, 99
				e.ArgumentsDelta[0] = '['
			case ToolUseDone:
				*e.Index, *e.OutputIndex = 99, 99
			case ProviderEvent:
				e.Raw[0] = '['
			}

		}}
	}), ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {

		return ctx, CallObserverFuncs{OnEventFunc: func(event Event) {

			if !reflect.DeepEqual(event, want[observed]) {
				t.Errorf("second hook saw mutated event %T: %#v", event, event)
			}
			observed++

		}}
	})))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	stream, err := client.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	defer stream.Close()
	for _, expected := range want {
		event, err := stream.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(event, expected) {
			t.Errorf("caller saw mutated event %T: %#v", event, event)
		}
	}
	if !reflect.DeepEqual(events, want) {
		t.Error("provider events were mutated")
	}
}

func TestStreamTextReturnsCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	client, err := New(&testProvider{
		name: "stream",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &closeErrStream{
				events: []Event{ContentDelta{Text: "ok"}, DoneEvent{FinishReason: FinishReasonStop, Provider: "stream", Model: req.Model}},
				err:    closeErr,
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.StreamText(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}}, nil)
	if !errors.Is(err, closeErr) {
		t.Fatalf("StreamText err = %v, want close error", err)
	}
}

func TestStreamTextPreservesHandleErrorOverCloseError(t *testing.T) {
	boom := errors.New("boom")
	closeErr := errors.New("close failed")
	client, err := New(&testProvider{
		name: "stream",
		streamFunc: func(ctx context.Context, req *Request) (Stream, error) {
			return &closeErrStream{
				events: []Event{ContentDelta{Text: "partial"}},
				err:    closeErr,
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = client.StreamText(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}}, func(string) error {
		return boom
	})
	if !errors.Is(err, boom) {
		t.Fatalf("StreamText err = %v, want callback error", err)
	}
}

type testStreamWithError struct {
	events []Event
	index  int
	err    error
}

func (s *testStreamWithError) Next() (Event, error) {
	if s.index >= len(s.events) {
		return nil, s.err
	}
	event := s.events[s.index]
	s.index++
	return event, nil
}

func (s *testStreamWithError) Close() error { return nil }

type closeErrStream struct {
	events []Event
	index  int
	err    error
}

func (s *closeErrStream) Next() (Event, error) {
	if s.index >= len(s.events) {
		return nil, io.EOF
	}
	event := s.events[s.index]
	s.index++
	return event, nil
}

func (s *closeErrStream) Close() error { return s.err }

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
	if err := options.Set("invalid", func() {}); !IsValidationError(err) {
		t.Fatalf("encoding error = %v", err)
	}
	if _, exists := options["invalid"]; exists {
		t.Fatal("failed Set changed options")
	}
	for _, raw := range []json.RawMessage{nil, json.RawMessage(`{"unterminated":`), json.RawMessage(`1 2`), {'"', 0xff, '"'}} {
		if _, err := (ProviderOptions{"invalid": raw}).Decode(); !IsValidationError(err) {
			t.Fatalf("raw %q: %v", raw, err)
		}
	}
}

func TestClientOwnsDefaults(t *testing.T) {
	max := 10
	option := WithDefaults(RequestDefaults{MaxTokens: &max})
	provider := &testProvider{name: "test"}
	client, err := New(provider, option)
	if err != nil {
		t.Fatal(err)
	}
	max = 99
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}}); err != nil {
		t.Fatal(err)
	}
	if *provider.lastReq.MaxTokens != 10 {
		t.Fatal("Client shares caller defaults")
	}
}

func TestUsageSnapshotsAndKnownZero(t *testing.T) {
	if (Usage{}).HasTokens() {
		t.Fatal("unknown usage has tokens")
	}
	usage := Usage{InputTokens: IntPtr(0), OutputTokens: IntPtr(2)}
	if !usage.HasTokens() {
		t.Fatal("known zero is unknown")
	}
	collector := NewEventCollector()
	if _, err := collector.Apply(UsageEvent{Usage: usage}); err != nil {
		t.Fatal(err)
	}
	*usage.InputTokens = 9
	first := collector.Response()
	if *first.Usage.InputTokens != 0 {
		t.Fatal("collector shares incoming usage")
	}
	*first.Usage.InputTokens = 8
	if *collector.Response().Usage.InputTokens != 0 {
		t.Fatal("Response shares collector usage")
	}
	event := UsageEvent{Usage: Usage{InputTokens: IntPtr(0)}}
	copied := cloneEvent(event).(UsageEvent)
	*copied.Usage.InputTokens = 7
	if *event.Usage.InputTokens != 0 {
		t.Fatal("event snapshot shares usage")
	}
	original := &Response{Usage: event.Usage}
	*cloneResponse(original).Usage.InputTokens = 6
	if *original.Usage.InputTokens != 0 {
		t.Fatal("response snapshot shares usage")
	}
}

func TestValidateHistoryRejectsDuplicateResults(t *testing.T) {
	messages := []Message{Assistant(ToolUseBlock{ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}), ToolResultText("call", "ok"), ToolResultText("call", "again")}
	if err := ValidateHistory(messages); !IsValidationError(err) {
		t.Fatalf("duplicate result = %v", err)
	}
}
