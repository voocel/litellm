// Package gateway carries model calls over HTTP, so that an application can
// reach models through a gateway that holds the vendor keys and meters the
// usage: an agent in a sandbox, say, that must never see a key. [Provider]
// is the client, an ordinary litellm.Provider whose calls run on a [Server];
// the Server makes each call with the Client it routes the call to and
// streams the call back.
//
// A call POSTs its litellm.Request as JSON, and the reply streams the call's
// events as JSON lines up to done or error, with a heartbeat line whenever
// the upstream is silent for a while, so that proxies keep the connection
// open. Blocks keep their provider state, and errors keep their type, retry
// facts and upstream provider, so a call through a gateway behaves as one
// made to the vendor directly; only the Server's own vendor key being
// rejected is a provider error rather than an auth error, which would blame
// the caller's key. Both ends must run the same version of this package.
//
// A Server neither authenticates nor meters. Serve it behind your
// authentication, which puts the caller on the request context, and meter
// with a litellm.Observer on the routed Clients, which see that context.
package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/voocel/litellm"
)

// event is one line of a reply: a litellm.Event, or the error that ends the
// call.
type event struct {
	Type            string               `json:"type"`
	Index           int                  `json:"index,omitempty"`
	Block           *block               `json:"block,omitempty"`
	Text            string               `json:"text,omitempty"`
	Usage           *litellm.Usage       `json:"usage,omitempty"`
	Warning         *litellm.Warning     `json:"warning,omitempty"`
	Name            string               `json:"name,omitempty"`
	Raw             json.RawMessage      `json:"raw,omitempty"`
	FinishReason    litellm.FinishReason `json:"finish_reason,omitempty"`
	FinishReasonRaw string               `json:"finish_reason_raw,omitempty"`
	Provider        string               `json:"provider,omitempty"`
	Model           string               `json:"model,omitempty"`
	Error           *wireError           `json:"error,omitempty"`
}

type block struct{ litellm.Block }

func (b block) MarshalJSON() ([]byte, error) { return json.Marshal(b.Block) }

func (b *block) UnmarshalJSON(data []byte) (err error) {
	b.Block, err = litellm.UnmarshalBlock(data)
	return err
}

func toEvent(ev litellm.Event) event {
	switch e := ev.(type) {
	case litellm.BlockStart:
		return event{Type: "block_start", Index: e.Index, Block: &block{e.Block}}
	case litellm.TextDelta:
		return event{Type: "text_delta", Index: e.Index, Text: e.Text}
	case litellm.ReasoningDelta:
		return event{Type: "reasoning_delta", Index: e.Index, Text: e.Text}
	case litellm.ToolUseDelta:
		return event{Type: "tool_use_delta", Index: e.Index, Text: e.Arguments}
	case litellm.BlockEnd:
		return event{Type: "block_end", Index: e.Index, Block: &block{metadata(e.Block)}}
	case litellm.UsageEvent:
		return event{Type: "usage", Usage: &e.Usage}
	case litellm.WarningEvent:
		return event{Type: "warning", Warning: &e.Warning}
	case litellm.ProviderEvent:
		return event{Type: "provider_event", Name: e.Name, Raw: e.Raw}
	case litellm.DoneEvent:
		return event{Type: "done", FinishReason: e.FinishReason, FinishReasonRaw: e.FinishReasonRaw, Provider: e.Provider, Model: e.Model}
	}
	panic(fmt.Sprintf("gateway: unknown event %T", ev))
}

// metadata returns the completed block b without the content its deltas
// already carried.
func metadata(b litellm.Block) litellm.Block {
	switch b := b.(type) {
	case litellm.TextBlock:
		b.Text = ""
		return b
	case litellm.ReasoningBlock:
		b.Text = ""
		return b
	case litellm.ToolUseBlock:
		b.Arguments = ""
		return b
	}
	return b
}

// fromEvent returns the litellm.Event e carries, or the error it reports.
func fromEvent(e event) (litellm.Event, error) {
	switch e.Type {
	case "block_start", "block_end":
		if e.Block == nil {
			return nil, fmt.Errorf("%s without a block", e.Type)
		}
		if e.Type == "block_start" {
			return litellm.BlockStart{Index: e.Index, Block: e.Block.Block}, nil
		}
		return litellm.BlockEnd{Index: e.Index, Block: e.Block.Block}, nil
	case "text_delta":
		return litellm.TextDelta{Index: e.Index, Text: e.Text}, nil
	case "reasoning_delta":
		return litellm.ReasoningDelta{Index: e.Index, Text: e.Text}, nil
	case "tool_use_delta":
		return litellm.ToolUseDelta{Index: e.Index, Arguments: e.Text}, nil
	case "usage":
		if e.Usage == nil {
			return nil, errors.New("usage without the usage")
		}
		return litellm.UsageEvent{Usage: *e.Usage}, nil
	case "warning":
		if e.Warning == nil {
			return nil, errors.New("warning without the warning")
		}
		return litellm.WarningEvent{Warning: *e.Warning}, nil
	case "provider_event":
		return litellm.ProviderEvent{Name: e.Name, Raw: e.Raw}, nil
	case "done":
		return litellm.DoneEvent{FinishReason: e.FinishReason, FinishReasonRaw: e.FinishReasonRaw, Provider: e.Provider, Model: e.Model}, nil
	case "error":
		if e.Error == nil {
			return nil, errors.New("error without the error")
		}
		return nil, e.Error.toError()
	}
	return nil, fmt.Errorf("unknown event %q", e.Type)
}

// wireError is a litellm.Error without its cause.
type wireError struct {
	Type         litellm.ErrorType `json:"type"`
	Code         string            `json:"code,omitempty"`
	Message      string            `json:"message"`
	Provider     string            `json:"provider,omitempty"`
	StatusCode   int               `json:"status_code,omitempty"`
	Temporary    bool              `json:"temporary,omitempty"`
	RetryAfterMS int64             `json:"retry_after_ms,omitempty"`
}

// heartbeat is the type of the line the Server sends while the upstream is
// silent; the Provider skips it.
const heartbeat = "heartbeat"

// toWireError keeps what err tells a caller; an error that is not a
// litellm.Error is of fallback type.
func toWireError(err error, fallback litellm.ErrorType) *wireError {
	var e *litellm.Error
	if !errors.As(err, &e) {
		e = litellm.NewError("", fallback, err.Error(), nil)
	}
	message := e.Message
	if e.Cause != nil {
		if cause := e.Cause.Error(); !strings.Contains(message, cause) {
			message = strings.TrimPrefix(message+": "+cause, ": ")
		}
	}
	return &wireError{
		Type:         e.Type,
		Code:         e.Code,
		Message:      message,
		Provider:     e.Provider,
		StatusCode:   e.StatusCode,
		Temporary:    e.Temporary,
		RetryAfterMS: e.RetryAfter.Milliseconds(),
	}
}

func (w *wireError) toError() *litellm.Error {
	return &litellm.Error{
		Type:       w.Type,
		Code:       w.Code,
		Message:    w.Message,
		Provider:   w.Provider,
		StatusCode: w.StatusCode,
		Temporary:  w.Temporary,
		RetryAfter: time.Duration(w.RetryAfterMS) * time.Millisecond,
	}
}
