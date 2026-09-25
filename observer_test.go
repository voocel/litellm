package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"
)

// callObserver adapts functions to CallObserver; nil functions are skipped.
type callObserver struct {
	onEvent func(Event)
	end     func(CallResult)
}

func (o callObserver) OnEvent(e Event) {
	if o.onEvent != nil {
		o.onEvent(e)
	}
}

func (o callObserver) End(r CallResult) {
	if o.end != nil {
		o.end(r)
	}
}

func endObserver(end func(CallResult)) Observer {
	return ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
		return ctx, callObserver{end: end}
	})
}

func TestObserverContextChainAndEndOrder(t *testing.T) {
	type key int
	var order []int
	var infos []CallInfo
	makeObserver := func(id int) Observer {
		return ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
			if id == 2 && ctx.Value(key(1)) == nil {
				t.Fatal("second observer missed derived context")
			}
			infos = append(infos, info)
			ctx = context.WithValue(ctx, key(id), id)
			return ctx, callObserver{end: func(result CallResult) {
				order = append(order, id)
				if result.Status != CallCompleted || result.Duration <= 0 || result.Err != nil || result.Response.Text() != "ok" {
					t.Fatalf("result=%+v", result)
				}
			}}
		})
	}
	optOut := ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) { return ctx, nil })
	provider := &testProvider{name: "test", chatFunc: func(ctx context.Context, _ *Request) (*Response, error) {
		if ctx.Value(key(1)) == nil || ctx.Value(key(2)) == nil {
			t.Fatal("provider missed observer context")
		}
		return &Response{Blocks: []Block{Text("ok")}}, nil
	}}
	c, err := New(provider, WithObservers(makeObserver(1), nil, optOut, makeObserver(2)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = c.Chat(context.Background(), Request{Model: "m", Messages: hi}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []int{2, 1}) {
		t.Fatalf("end order=%v", order)
	}
	info := infos[0]
	if info.Provider != "test" || info.Streaming || info.StartedAt.IsZero() || info.Request.Model != "m" {
		t.Fatalf("info=%+v", info)
	}
	if infos[1].Request != info.Request || info.Request == provider.lastReq {
		t.Fatal("observers must share one request snapshot, separate from the provider's")
	}
}

func TestObserverSnapshotsAreSharedAndIsolated(t *testing.T) {
	t.Run("chat", func(t *testing.T) {
		var results []CallResult
		var warnings []Event
		observer := ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
			info.Request.Model = "mutated"
			info.Request.Messages[0].Blocks[0] = Text("mutated")
			return ctx, callObserver{
				onEvent: func(e Event) { warnings = append(warnings, e) },
				end: func(result CallResult) {
					results = append(results, result)
					result.Response.Blocks[0] = Text("mutated")
					result.Response.Warnings[0].Code = "mutated"
				},
			}
		})
		provider := &testProvider{name: "test", chatFunc: func(_ context.Context, req *Request) (*Response, error) {
			if req.Model != "m" || req.Messages[0].Blocks[0].(TextBlock).Text != "hi" {
				t.Fatalf("provider saw observer mutation: %+v", req)
			}
			return &Response{Blocks: []Block{Text("ok")}, Warnings: []Warning{{Code: "w"}}}, nil
		}}
		c, err := New(provider, WithObservers(observer, observer))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.Chat(context.Background(), Request{Model: "m", Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() != "ok" || resp.Warnings[0].Code != "w" {
			t.Fatalf("caller saw observer mutation: %#v", resp)
		}
		if results[0].Response != results[1].Response {
			t.Fatal("observers must share one result snapshot")
		}
		want := WarningEvent{Warning: Warning{Code: "w", Provider: "test"}}
		if len(warnings) != 2 || warnings[0] != want {
			t.Fatalf("warning events = %#v", warnings)
		}
	})

	t.Run("stream", func(t *testing.T) {
		newEvents := func() []Event {
			return []Event{
				BlockStart{Index: 0, Block: ReasoningBlock{State: testState(`{}`)}},
				ReasoningDelta{Index: 0, Text: "think"},
				BlockEnd{Index: 0},
				BlockStart{Index: 1, Block: TextBlock{}},
				TextDelta{Index: 1, Text: "hi"},
				BlockEnd{Index: 1, Block: TextBlock{Annotations: []Annotation{{Extra: json.RawMessage(`{}`)}}}},
				UsageEvent{Usage: Usage{InputTokens: new(1)}},
				ProviderEvent{Name: "vendor.event", Raw: json.RawMessage(`{"ok":true}`)},
				DoneEvent{FinishReason: FinishReasonStop, Provider: "test", Model: "m"},
			}
		}
		events := newEvents()
		var seen [2][]Event
		observer := func(i int) Observer {
			return ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
				return ctx, callObserver{onEvent: func(e Event) { seen[i] = append(seen[i], e) }}
			})
		}
		mutate := ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
			return ctx, callObserver{onEvent: func(event Event) {
				switch e := event.(type) {
				case BlockStart:
					if b, ok := e.Block.(ReasoningBlock); ok {
						b.State.Data[0] = '['
					}
				case BlockEnd:
					if b, ok := e.Block.(TextBlock); ok {
						b.Annotations[0].Extra[0] = '['
					}
				case UsageEvent:
					*e.Usage.InputTokens = 9
				case ProviderEvent:
					e.Raw[0] = '['
				}
			}}
		})
		c, err := New(&testProvider{name: "test", streamFunc: func(context.Context, *Request) (Stream, error) {
			return &testStream{events: events}, nil
		}}, WithObservers(observer(0), mutate, observer(1)))
		if err != nil {
			t.Fatal(err)
		}
		stream, err := c.Stream(context.Background(), Request{Model: "m", Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		var delivered []Event
		resp, err := Handle(stream, func(e Event) error {
			delivered = append(delivered, e)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(events, newEvents()) {
			t.Error("provider events were mutated")
		}
		raw := func(e Event) *byte { return &e.(ProviderEvent).Raw[0] }
		if raw(seen[0][7]) != raw(seen[1][7]) || raw(seen[0][7]) == raw(delivered[7]) {
			t.Error("observers must share one event snapshot, separate from the caller's")
		}
		if string(delivered[7].(ProviderEvent).Raw) != `{"ok":true}` || *delivered[6].(UsageEvent).Usage.InputTokens != 1 {
			t.Errorf("caller saw observer mutation: %#v", delivered)
		}
		reasoning := resp.Blocks[0].(ReasoningBlock)
		text := resp.Blocks[1].(TextBlock)
		if string(reasoning.State.Data) != `{}` || string(text.Annotations[0].Extra) != `{}` || *resp.Usage.InputTokens != 1 {
			t.Errorf("response saw observer mutation: %#v", resp)
		}
	})
}

func TestObserverSeesLocalValidationFailure(t *testing.T) {
	var results []CallResult
	c, _ := New(&testProvider{name: "test", chatFunc: func(context.Context, *Request) (*Response, error) {
		t.Fatal("invalid request reached provider")
		return nil, nil
	}}, WithObservers(endObserver(func(r CallResult) { results = append(results, r) })))
	if _, err := c.Chat(context.Background(), Request{}); !IsValidationError(err) {
		t.Fatalf("error=%v", err)
	}
	if _, err := c.Stream(context.Background(), Request{}); !IsValidationError(err) {
		t.Fatalf("error=%v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results=%v", results)
	}
	for _, r := range results {
		if r.Status != CallFailed || r.Response != nil || r.Err == nil {
			t.Fatalf("result=%+v", r)
		}
	}
}

func TestObserverStreamingTerminalStates(t *testing.T) {
	boom := errors.New("boom")
	for _, tc := range []struct {
		name    string
		want    CallStatus
		wantErr error
	}{
		{"done", CallCompleted, nil}, {"setup", CallFailed, boom}, {"runtime", CallFailed, boom},
		{"cancel", CallCanceled, context.Canceled}, {"cancel_close", CallCanceled, context.Canceled},
		{"deadline", CallFailed, context.DeadlineExceeded}, {"early_close", CallClosed, nil}, {"close_error", CallFailed, boom},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ends := 0
			var result CallResult
			var providerCtx context.Context
			p := &testProvider{name: "test", streamFunc: func(ctx context.Context, _ *Request) (Stream, error) {
				providerCtx = ctx
				switch tc.name {
				case "setup":
					return nil, boom
				case "cancel", "cancel_close":
					return blockingStream{ctx: ctx}, nil
				case "runtime":
					return &testStream{events: textEvents(0, "partial")[:2], err: boom}, nil
				case "deadline":
					return &testStream{err: context.DeadlineExceeded}, nil
				case "close_error":
					return &testStream{closeErr: boom}, nil
				default:
					return &testStream{events: append(textEvents(0, "ok"), DoneEvent{Provider: "test", Model: "m"})}, nil
				}
			}}
			c, _ := New(p, WithObservers(endObserver(func(r CallResult) { ends++; result = r })))
			s, err := c.Stream(ctx, Request{Model: "m", Messages: hi})
			if tc.name == "setup" {
				if !errors.Is(err, boom) {
					t.Fatal(err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if ends != 0 {
					t.Fatal("ended at stream creation")
				}
				switch tc.name {
				case "early_close", "close_error":
					_ = s.Close()
				case "cancel_close":
					cancel()
					_ = s.Close()
				default:
					if tc.name == "cancel" {
						cancel()
					}
					_, _ = Collect(s)
				}
				_ = s.Close()
				_ = s.Close()
				if _, err := s.Next(); err != io.EOF {
					t.Fatalf("Next after end=%v", err)
				}
			}
			if ends != 1 || result.Status != tc.want || !errors.Is(result.Err, tc.wantErr) {
				t.Fatalf("ends=%d result=%+v", ends, result)
			}
			if tc.name == "runtime" && (result.Response == nil || result.Response.Text() != "partial") {
				t.Fatalf("partial=%#v", result.Response)
			}
			if providerCtx.Err() == nil {
				t.Fatal("stream context leaked after termination")
			}
		})
	}
}

func TestObserverProtocolFailureDoesNotPublishDone(t *testing.T) {
	var result CallResult
	done := false
	c, _ := New(&testProvider{name: "test", streamFunc: func(context.Context, *Request) (Stream, error) {
		// The block is never ended, so Done violates the lifecycle.
		return &testStream{events: append(textEvents(0, "partial")[:2], DoneEvent{Provider: "test", Model: "m"})}, nil
	}}, WithObservers(ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
		return ctx, callObserver{
			onEvent: func(e Event) {
				if _, ok := e.(DoneEvent); ok {
					done = true
				}
			},
			end: func(r CallResult) { result = r },
		}
	})))
	s, err := c.Stream(context.Background(), Request{Model: "m", Messages: hi})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	_, err = Collect(s)
	if err == nil || done || result.Status != CallFailed || result.Response.Text() != "partial" {
		t.Fatalf("error=%v done=%v result=%+v", err, done, result)
	}
}

func TestObserverIdleTimeoutOnClose(t *testing.T) {
	for _, closeErr := range []error{nil, context.Canceled} {
		var result CallResult
		p := &testProvider{name: "test", streamFunc: func(context.Context, *Request) (Stream, error) { return &testStream{closeErr: closeErr}, nil }}
		c, _ := New(p, WithStreamIdleTimeout(time.Hour), WithObservers(endObserver(func(r CallResult) { result = r })))
		s, err := c.Stream(context.Background(), Request{Model: "m", Messages: hi})
		if err != nil {
			t.Fatal(err)
		}
		s.(*observedStream).inner.(*streamIdleWatchdog).fire()
		if err = s.Close(); !IsStreamIdleError(err) || result.Status != CallFailed || !IsStreamIdleError(result.Err) {
			t.Fatalf("close=%v result=%+v", err, result)
		}
	}
}
