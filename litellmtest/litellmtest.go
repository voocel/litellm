// Package litellmtest provides a scripted Provider for testing code that
// calls models, without a network or a vendor.
package litellmtest

import (
	"context"
	"fmt"
	"io"
	"slices"
	"sync"

	"github.com/voocel/litellm"
)

// Reply is one scripted call: the response it streams, or the error it fails
// with before streaming.
type Reply struct {
	Blocks []litellm.Block
	Usage  litellm.Usage
	// FinishReason defaults to tool_calls when Blocks hold a tool use, and to
	// stop otherwise.
	FinishReason litellm.FinishReason
	Err          error
	// StreamErr, if set, fails the stream with it once Blocks streamed,
	// instead of finishing it.
	StreamErr error
	// Stall, if set, holds the stream open once Blocks streamed, until the
	// call is cancelled.
	Stall bool
}

// Text replies with text.
func Text(text string) Reply { return Reply{Blocks: []litellm.Block{litellm.Text(text)}} }

// Respond replies with blocks.
func Respond(blocks ...litellm.Block) Reply { return Reply{Blocks: blocks} }

// Fail fails the call with err.
func Fail(err error) Reply { return Reply{Err: err} }

// Provider answers each call with the next of its replies, and records the
// requests it was called with. It is safe for concurrent use. A call beyond
// the script fails.
type Provider struct {
	mu       sync.Mutex
	replies  []Reply
	requests []litellm.Request
}

// New returns a Provider that replies in order.
func New(replies ...Reply) *Provider { return &Provider{replies: replies} }

// Name returns "test".
func (p *Provider) Name() string { return "test" }

// Requests returns the requests made so far, in order.
func (p *Provider) Requests() []litellm.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.requests)
}

func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	stream, err := p.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	return litellm.Collect(stream)
}

func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	p.mu.Lock()
	n := len(p.requests)
	p.requests = append(p.requests, *req)
	p.mu.Unlock()
	if n >= len(p.replies) {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, fmt.Sprintf("no reply scripted for call %d", n+1), nil)
	}
	reply := p.replies[n]
	if reply.Err != nil {
		return nil, reply.Err
	}
	return &stream{ctx: ctx, events: events(reply, req.Model), err: reply.StreamErr, stall: reply.Stall}, nil
}

// events streams reply the way a vendor would: each block opened, grown by
// one delta and closed with its provider state, then usage and done.
func events(reply Reply, model string) []litellm.Event {
	var out []litellm.Event
	finish := litellm.FinishReasonStop
	for i, block := range reply.Blocks {
		switch b := block.(type) {
		case litellm.TextBlock:
			out = append(out, litellm.BlockStart{Index: i, Block: litellm.TextBlock{}}, litellm.TextDelta{Index: i, Text: b.Text})
			out = append(out, litellm.BlockEnd{Index: i, Block: litellm.TextBlock{Annotations: b.Annotations, State: b.State}})
		case litellm.ReasoningBlock:
			out = append(out, litellm.BlockStart{Index: i, Block: litellm.ReasoningBlock{Summary: b.Summary}}, litellm.ReasoningDelta{Index: i, Text: b.Text})
			out = append(out, litellm.BlockEnd{Index: i, Block: litellm.ReasoningBlock{State: b.State}})
		case litellm.ToolUseBlock:
			finish = litellm.FinishReasonToolCall
			out = append(out, litellm.BlockStart{Index: i, Block: litellm.ToolUseBlock{ID: b.ID, Name: b.Name}}, litellm.ToolUseDelta{Index: i, Arguments: b.Arguments})
			out = append(out, litellm.BlockEnd{Index: i, Block: litellm.ToolUseBlock{State: b.State}})
		default:
			panic(fmt.Sprintf("litellmtest: a reply cannot hold %T", block))
		}
	}
	if reply.StreamErr != nil || reply.Stall {
		return out
	}
	if reply.FinishReason != "" {
		finish = reply.FinishReason
	}
	if reply.Usage != (litellm.Usage{}) {
		out = append(out, litellm.UsageEvent{Usage: reply.Usage})
	}
	return append(out, litellm.DoneEvent{FinishReason: finish, Provider: "test", Model: model})
}

type stream struct {
	ctx    context.Context
	events []litellm.Event
	err    error
	stall  bool
}

func (s *stream) Next() (litellm.Event, error) {
	if err := s.ctx.Err(); err != nil {
		return nil, err
	}
	if len(s.events) == 0 {
		switch {
		case s.err != nil:
			return nil, s.err
		case s.stall:
			<-s.ctx.Done()
			return nil, s.ctx.Err()
		}
		return nil, io.EOF
	}
	ev := s.events[0]
	s.events = s.events[1:]
	return ev, nil
}

func (s *stream) Close() error { return nil }
