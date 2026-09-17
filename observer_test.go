package litellm

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"
)

func TestObserverContextChainAndEndOrder(t *testing.T) {
	type key int
	var order []int
	makeObserver := func(id int) Observer {
		return ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
			if id == 2 && ctx.Value(key(1)) != "first" {
				t.Fatal("second observer missed derived context")
			}
			if info.Request.Model != "m" || info.StartedAt.IsZero() {
				t.Fatalf("info=%+v", info)
			}
			ctx = context.WithValue(ctx, key(id), "first")
			return ctx, CallObserverFuncs{EndFunc: func(result CallResult) {
				order = append(order, id)
				if result.Status != CallCompleted || result.Duration <= 0 || result.Err != nil {
					t.Fatalf("result=%+v", result)
				}
			}}
		})
	}
	provider := &testProvider{name: "test", chatFunc: func(ctx context.Context, _ *Request) (*Response, error) {
		if ctx.Value(key(1)) == nil || ctx.Value(key(2)) == nil {
			t.Fatal("provider missed observer context")
		}
		return &Response{Blocks: []Block{Text("ok")}}, nil
	}}
	c, err := New(provider, WithObservers(makeObserver(1), makeObserver(2)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = c.Chat(context.Background(), Request{Model: "m", Messages: []Message{UserText("hello")}}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(order, []int{2, 1}) {
		t.Fatalf("end order=%v", order)
	}
}

func TestObserverSeesLocalValidationFailure(t *testing.T) {
	var results []CallResult
	c, _ := New(&testProvider{name: "test", chatFunc: func(context.Context, *Request) (*Response, error) {
		t.Fatal("invalid request reached provider")
		return nil, nil
	}}, WithObservers(ObserverFunc(func(ctx context.Context, info CallInfo) (context.Context, CallObserver) {
		if info.Request.Model != "" {
			t.Fatal("unexpected request")
		}
		return ctx, CallObserverFuncs{EndFunc: func(r CallResult) { results = append(results, r) }}
	})))
	for _, streaming := range []bool{false, true} {
		var err error
		if streaming {
			_, err = c.Stream(context.Background(), Request{})
		} else {
			_, err = c.Chat(context.Background(), Request{})
		}
		if !IsValidationError(err) {
			t.Fatalf("error=%v", err)
		}
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
					return &testStreamWithError{events: []Event{ContentDelta{Text: "partial"}}, err: boom}, nil
				case "deadline":
					return &testStreamWithError{err: context.DeadlineExceeded}, nil
				case "close_error":
					return &closeErrStream{err: boom}, nil
				default:
					return &testStream{events: []Event{ContentDelta{Text: "ok"}, DoneEvent{Provider: "test", Model: "m"}}}, nil
				}
			}}
			c, _ := New(p, WithObservers(ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
				return ctx, CallObserverFuncs{EndFunc: func(r CallResult) { ends++; result = r }}
			})))
			s, err := c.Stream(ctx, Request{Model: "m", Messages: []Message{UserText("hello")}})
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
		return &testStream{events: []Event{ContentStart{Block: TextBlock{Text: "partial"}, ContentIndex: IntPtr(0)}, DoneEvent{Provider: "test", Model: "m"}}}, nil
	}}, WithObservers(ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
		return ctx, CallObserverFuncs{
			OnEventFunc: func(e Event) {
				if _, ok := e.(DoneEvent); ok {
					done = true
				}
			}, EndFunc: func(r CallResult) { result = r },
		}
	})))
	s, err := c.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
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
		p := &testProvider{name: "test", streamFunc: func(context.Context, *Request) (Stream, error) { return &closeErrStream{err: closeErr}, nil }}
		c, _ := New(p, WithStreamIdleTimeout(time.Hour), WithObservers(ObserverFunc(func(ctx context.Context, _ CallInfo) (context.Context, CallObserver) {
			return ctx, CallObserverFuncs{EndFunc: func(r CallResult) { result = r }}
		})))
		s, err := c.Stream(context.Background(), Request{Model: "m", Messages: []Message{UserText("hi")}})
		if err != nil {
			t.Fatal(err)
		}
		s.(*observedStream).inner.(*streamIdleWatchdog).fire()
		if err = s.Close(); !IsStreamIdleError(err) || result.Status != CallFailed || !IsStreamIdleError(result.Err) {
			t.Fatalf("close=%v result=%+v", err, result)
		}
	}
}
