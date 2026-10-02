package gateway_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/gateway"
	"github.com/voocel/litellm/litellmtest"
)

// serve runs a Server that routes "smart" to upstream as "vendor-model",
// and returns a Client calling it with key.
func serve(t *testing.T, upstream litellm.Provider, opts ...litellm.ClientOption) (*litellm.Client, *http.Request) {
	t.Helper()
	routed, err := litellm.New(upstream, opts...)
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var last http.Request
	srv := httptest.NewServer(&gateway.Server{Route: func(r *http.Request, req *litellm.Request) (*litellm.Client, error) {
		mu.Lock()
		last = *r
		mu.Unlock()
		if req.Model != "smart" {
			return nil, errors.New("no such model")
		}
		req.Model = "vendor-model"
		return routed, nil
	}})
	t.Cleanup(srv.Close)
	p, err := gateway.New(gateway.Config{BaseURL: srv.URL, APIKey: "team-token"})
	if err != nil {
		t.Fatal(err)
	}
	client, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	return client, &last
}

func ask(model string) litellm.Request {
	return litellm.Request{Model: model, Messages: []litellm.Message{litellm.UserText("hi")}}
}

// A call through the gateway replies as the vendor would: blocks with their
// provider state, usage, and the provider and model that made it.
func TestCallThroughTheGateway(t *testing.T) {
	state := &litellm.ProviderState{Provider: "test", Model: "vendor-model", Data: json.RawMessage(`{"signature":"s"}`)}
	blocks := []litellm.Block{
		litellm.ReasoningBlock{Text: "let me look", State: state},
		litellm.TextBlock{Text: "reading", State: state},
		litellm.ToolUseBlock{ID: "c1", Name: "read", Arguments: `{"path":"a.go"}`, State: state},
	}
	upstream := litellmtest.New(litellmtest.Reply{Blocks: blocks, Usage: litellm.Usage{InputTokens: 10, OutputTokens: 4}})
	client, _ := serve(t, upstream)

	stream, err := client.Stream(context.Background(), ask("smart"))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var deltas []string
	resp, err := litellm.Handle(stream, func(ev litellm.Event) error {
		if d, ok := ev.(litellm.TextDelta); ok {
			deltas = append(deltas, d.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(resp.Blocks, blocks) {
		t.Fatalf("blocks = %#v", resp.Blocks)
	}
	if resp.Provider != "test" || resp.Model != "vendor-model" || resp.FinishReason != litellm.FinishReasonToolCall ||
		resp.Usage != (litellm.Usage{InputTokens: 10, OutputTokens: 4}) {
		t.Fatalf("response = %#v", resp)
	}
	if !reflect.DeepEqual(deltas, []string{"reading"}) {
		t.Fatalf("text deltas = %v", deltas)
	}
}

// The request reaches the vendor as the caller made it, but for the model the
// gateway routed it to; the caller's key reaches only the gateway.
func TestRequestCrossesWhole(t *testing.T) {
	upstream := litellmtest.New(litellmtest.Text("ok"))
	client, seen := serve(t, upstream)
	maxTokens := 256
	options, err := litellm.NewProviderOptions(map[string]any{"prompt_cache_key": "conv-1"})
	if err != nil {
		t.Fatal(err)
	}
	state := &litellm.ProviderState{Provider: "test", Data: json.RawMessage(`{"s":1}`)}
	req := litellm.Request{
		Model: "smart",
		Messages: []litellm.Message{
			litellm.System("be brief"),
			litellm.UserText("read a.go"),
			litellm.Assistant(litellm.ReasoningBlock{Text: "hm", State: state}, litellm.ToolUseBlock{ID: "c1", Name: "read", Arguments: `{"path":"a.go"}`}),
			litellm.ToolResultText("c1", "package a"),
		},
		MaxTokens:       &maxTokens,
		Tools:           []litellm.Tool{{Name: "read", Description: "Read a file", Parameters: litellm.Schema(`{"type":"object"}`)}},
		ToolChoice:      &litellm.ToolChoice{Mode: litellm.ToolChoiceAuto},
		Thinking:        &litellm.Thinking{Effort: "high"},
		ProviderOptions: options,
	}
	if _, err := client.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	got := upstream.Requests()[0]
	want := req
	want.Model = "vendor-model"
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("vendor got\n%#v\nwant\n%#v", got, want)
	}
	if seen.Header.Get("Authorization") != "Bearer team-token" {
		t.Fatalf("gateway saw headers %v", seen.Header)
	}
}

// Errors keep what callers act on: type, retry facts and the provider.
func TestErrorsKeepTheirFacts(t *testing.T) {
	limited := litellm.NewError("test", litellm.ErrorTypeRateLimit, "slow down", nil)
	limited.StatusCode, limited.RetryAfter = http.StatusTooManyRequests, 3*time.Second
	overflow := litellm.NewError("test", litellm.ErrorTypeContextOverflow, "prompt is too long", nil)
	client, _ := serve(t, litellmtest.New(litellmtest.Fail(limited), litellmtest.Fail(overflow)))

	_, err := client.Stream(context.Background(), ask("smart"))
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit || !litellm.IsTemporaryError(err) || litellm.RetryAfter(err) != 3*time.Second ||
		!strings.HasPrefix(err.Error(), "test: ") {
		t.Fatalf("rate limit came back as %v", err)
	}
	_, err = client.Chat(context.Background(), ask("smart"))
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeContextOverflow || litellm.IsTemporaryError(err) {
		t.Fatalf("overflow came back as %v", err)
	}
}

// A call that fails while streaming ends with the error, after the content
// streamed before it.
func TestErrorMidStream(t *testing.T) {
	client, _ := serve(t, &failing{err: litellm.NewError("test", litellm.ErrorTypeOverloaded, "busy", nil)})
	stream, err := client.Stream(context.Background(), ask("smart"))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	resp, err := litellm.Collect(stream)
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeOverloaded || !litellm.IsTemporaryError(err) {
		t.Fatalf("err = %v", err)
	}
	if resp.Text() != "partial" {
		t.Fatalf("partial response = %#v", resp)
	}
}

// Refusals come back typed: the gateway's own, and those of anything in
// front of it.
func TestRefusals(t *testing.T) {
	client, _ := serve(t, litellmtest.New())
	if _, err := client.Chat(context.Background(), ask("unknown")); litellm.ErrorTypeOf(err) != litellm.ErrorTypeModel || !strings.Contains(err.Error(), "no such model") {
		t.Fatalf("unknown model = %v", err)
	}

	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		http.Error(w, "too many requests", http.StatusTooManyRequests)
	}))
	defer front.Close()
	p, _ := gateway.New(gateway.Config{BaseURL: front.URL})
	direct, _ := litellm.New(p)
	_, err := direct.Chat(context.Background(), ask("smart"))
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeRateLimit || litellm.RetryAfter(err) != 7*time.Second {
		t.Fatalf("429 in front = %v", err)
	}
}

// Cancelling a call cancels the vendor call it runs on.
func TestCancelReachesTheVendor(t *testing.T) {
	upstream := &blocking{started: make(chan struct{}), stopped: make(chan error, 1)}
	client, _ := serve(t, upstream)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := client.Chat(ctx, ask("smart"))
		done <- err
	}()
	<-upstream.started
	cancel()
	if err := <-done; err == nil {
		t.Fatal("cancelled call succeeded")
	}
	select {
	case err := <-upstream.stopped:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("vendor call ended with %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("vendor call outlived the cancelled call")
	}
}

type callerKey struct{}

// An observer on the routed Client meters each call against the caller the
// gateway's authentication put on the request.
func TestMeterByCaller(t *testing.T) {
	var mu sync.Mutex
	billed := map[string]int{}
	meter := litellm.ObserverFunc(func(ctx context.Context, _ litellm.CallInfo) (context.Context, litellm.CallObserver) {
		return ctx, endFunc(func(res litellm.CallResult) {
			mu.Lock()
			billed[ctx.Value(callerKey{}).(string)] += res.Response.Usage.InputTokens
			mu.Unlock()
		})
	})
	routed, _ := litellm.New(litellmtest.New(
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("a")}, Usage: litellm.Usage{InputTokens: 5}},
		litellmtest.Reply{Blocks: []litellm.Block{litellm.Text("b")}, Usage: litellm.Usage{InputTokens: 5}},
	), litellm.WithObservers(meter))
	gw := &gateway.Server{Route: func(*http.Request, *litellm.Request) (*litellm.Client, error) { return routed, nil }}
	tokens := map[string]string{"t-ann": "ann", "t-bob": "bob"}
	auth := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		caller, ok := tokens[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
		if !ok {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		gw.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), callerKey{}, caller)))
	})
	srv := httptest.NewServer(auth)
	defer srv.Close()

	for _, token := range []string{"t-ann", "t-bob", "t-eve"} {
		p, _ := gateway.New(gateway.Config{BaseURL: srv.URL, APIKey: token})
		client, _ := litellm.New(p)
		_, err := client.Chat(context.Background(), ask("m"))
		if (token == "t-eve") != (litellm.ErrorTypeOf(err) == litellm.ErrorTypeAuth) {
			t.Fatalf("%s: %v", token, err)
		}
	}
	if !reflect.DeepEqual(billed, map[string]int{"ann": 5, "bob": 5}) {
		t.Fatalf("billed = %v", billed)
	}
}

// Malformed tool arguments cross the gateway as the model wrote them, and the
// caller's Client warns about them once.
func TestMalformedArgumentsWarnOnce(t *testing.T) {
	call := litellm.ToolUseBlock{ID: "c1", Name: "read", Arguments: `{"path":`}
	client, _ := serve(t, litellmtest.New(litellmtest.Reply{Blocks: []litellm.Block{call}}))
	resp, err := client.Chat(context.Background(), ask("smart"))
	if err != nil {
		t.Fatal(err)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].Arguments != `{"path":` {
		t.Fatalf("tool calls = %#v", calls)
	}
	if len(resp.Warnings) != 1 || resp.Warnings[0].Code != "litellm.tool_arguments_invalid" {
		t.Fatalf("warnings = %+v", resp.Warnings)
	}
}

// The upstream rejecting the gateway's vendor key is not the caller's auth
// failure.
func TestUpstreamKeyRejected(t *testing.T) {
	rejected := litellm.NewError("test", litellm.ErrorTypeAuth, "invalid x-api-key", nil)
	rejected.StatusCode = http.StatusUnauthorized
	client, _ := serve(t, litellmtest.New(litellmtest.Fail(rejected)))
	_, err := client.Chat(context.Background(), ask("smart"))
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeProvider || litellm.IsTemporaryError(err) || !strings.Contains(err.Error(), "upstream key rejected: invalid x-api-key") {
		t.Fatalf("err = %v", err)
	}

	client, _ = serve(t, &failing{err: rejected})
	stream, err := client.Stream(context.Background(), ask("smart"))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := litellm.Collect(stream); litellm.ErrorTypeOf(err) != litellm.ErrorTypeProvider || !strings.Contains(err.Error(), "upstream key rejected") {
		t.Fatalf("mid-stream err = %v", err)
	}
}

// A body over MaxRequestBytes is refused with 413.
func TestRequestBodyCap(t *testing.T) {
	srv := httptest.NewServer(&gateway.Server{Route: func(*http.Request, *litellm.Request) (*litellm.Client, error) {
		t.Error("an oversized call was routed")
		return nil, errors.New("unreachable")
	}})
	defer srv.Close()
	body := io.MultiReader(strings.NewReader(`{"model":"`), io.LimitReader(repeat('a'), gateway.MaxRequestBytes))
	resp, err := http.Post(srv.URL, "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

type repeat byte

func (r repeat) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(r)
	}
	return len(p), nil
}

// While the upstream is silent the Server sends heartbeats, which the
// Provider skips.
func TestHeartbeat(t *testing.T) {
	gateway.SetHeartbeatInterval(t, 5*time.Millisecond)
	client, _ := serve(t, &slow{delay: 50 * time.Millisecond})
	resp, err := client.Chat(context.Background(), ask("smart"))
	if err != nil || resp.Text() != "ok" {
		t.Fatalf("resp = %#v, err = %v", resp, err)
	}

	routed, _ := litellm.New(&slow{delay: 50 * time.Millisecond})
	srv := httptest.NewServer(&gateway.Server{Route: func(*http.Request, *litellm.Request) (*litellm.Client, error) { return routed, nil }})
	t.Cleanup(srv.Close)
	reply, err := http.Post(srv.URL, "application/json", strings.NewReader(`{"model":"m","messages":[{"role":"user","blocks":[{"type":"text","text":"hi"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer reply.Body.Close()
	data, _ := io.ReadAll(reply.Body)
	if first, _, _ := strings.Cut(string(data), "\n"); first != `{"type":"heartbeat"}` {
		t.Fatalf("reply = %s", data)
	}
}

// slow replies "ok" after delay.
type slow struct{ delay time.Duration }

func (*slow) Name() string { return "test" }
func (s *slow) Chat(context.Context, *litellm.Request) (*litellm.Response, error) {
	return nil, errors.New("not used")
}
func (s *slow) Stream(context.Context, *litellm.Request) (litellm.Stream, error) {
	return &delayed{delay: s.delay, script: script{events: []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 0, Text: "ok"},
		litellm.BlockEnd{Index: 0, Block: litellm.TextBlock{}},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonStop, Provider: "test", Model: "m"},
	}}}, nil
}

// delayed waits delay before its first event.
type delayed struct {
	delay time.Duration
	script
}

func (d *delayed) Next() (litellm.Event, error) {
	time.Sleep(d.delay)
	d.delay = 0
	return d.script.Next()
}

type endFunc func(litellm.CallResult)

func (endFunc) OnEvent(litellm.Event)        {}
func (f endFunc) End(res litellm.CallResult) { f(res) }

// failing streams some text, then fails with err.
type failing struct{ err error }

func (*failing) Name() string { return "test" }
func (f *failing) Chat(context.Context, *litellm.Request) (*litellm.Response, error) {
	return nil, errors.New("not used")
}
func (f *failing) Stream(context.Context, *litellm.Request) (litellm.Stream, error) {
	return &script{events: []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
		litellm.TextDelta{Index: 0, Text: "partial"},
	}, err: f.err}, nil
}

type script struct {
	events []litellm.Event
	err    error
}

func (s *script) Next() (litellm.Event, error) {
	if len(s.events) == 0 {
		return nil, s.err
	}
	ev := s.events[0]
	s.events = s.events[1:]
	return ev, nil
}
func (s *script) Close() error { return nil }

// blocking streams nothing until its call is cancelled.
type blocking struct {
	started chan struct{}
	stopped chan error
}

func (*blocking) Name() string { return "test" }
func (b *blocking) Chat(context.Context, *litellm.Request) (*litellm.Response, error) {
	return nil, errors.New("not used")
}
func (b *blocking) Stream(ctx context.Context, _ *litellm.Request) (litellm.Stream, error) {
	close(b.started)
	return &waiting{ctx: ctx, stopped: b.stopped}, nil
}

type waiting struct {
	ctx     context.Context
	stopped chan error
}

func (w *waiting) Next() (litellm.Event, error) {
	<-w.ctx.Done()
	w.stopped <- w.ctx.Err()
	return nil, w.ctx.Err()
}
func (w *waiting) Close() error { return nil }
