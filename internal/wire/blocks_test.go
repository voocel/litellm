package wire

import (
	"io"
	"reflect"
	"testing"

	"github.com/voocel/litellm"
)

func TestTracker(t *testing.T) {
	var tracker BlockTracker[string] // the zero value is ready to use
	var events []litellm.Event
	var index int

	events, index = tracker.Open(events, "text", litellm.TextBlock{})
	events, again := tracker.Open(events, "text", litellm.TextBlock{})
	events, tool := tracker.Open(events, "tool", litellm.ToolUseBlock{ID: "call", Name: "t"})
	events, reasoning := tracker.Open(events, "reasoning", litellm.ReasoningBlock{})
	if index != 0 || again != 0 || tool != 1 || reasoning != 2 || len(events) != 3 {
		t.Fatalf("indexes = %d %d %d %d, events = %d", index, again, tool, reasoning, len(events))
	}
	if i, ok := tracker.Index("tool"); !ok || i != 1 {
		t.Fatalf("Index(tool) = %d, %v", i, ok)
	}
	if _, ok := tracker.Index("missing"); ok {
		t.Fatal("Index reported an unopened key")
	}

	events = tracker.Close(events, "tool", litellm.ToolUseBlock{Signature: "sig"})
	events = tracker.Close(events, "tool", nil)
	if _, ok := tracker.Index("tool"); ok {
		t.Fatal("closed key is still open")
	}
	// CloseAll ends the rest in index order, with finals from the callback.
	events = tracker.CloseAll(events, func(key string) litellm.Block {
		if key == "reasoning" {
			return litellm.ReasoningBlock{Signature: "r"}
		}
		return nil
	})
	events = tracker.CloseAll(events, nil)
	// A closed key opens a new block.
	events, reopened := tracker.Open(events, "tool", litellm.ToolUseBlock{ID: "call_2", Name: "t"})
	events = tracker.CloseAll(events, nil)
	if reopened != 3 {
		t.Fatalf("reopened index = %d, want 3", reopened)
	}

	want := []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.TextBlock{}},
		litellm.BlockStart{Index: 1, Block: litellm.ToolUseBlock{ID: "call", Name: "t"}},
		litellm.BlockStart{Index: 2, Block: litellm.ReasoningBlock{}},
		litellm.BlockEnd{Index: 1, Block: litellm.ToolUseBlock{Signature: "sig"}},
		litellm.BlockEnd{Index: 0},
		litellm.BlockEnd{Index: 2, Block: litellm.ReasoningBlock{Signature: "r"}},
		litellm.BlockStart{Index: 3, Block: litellm.ToolUseBlock{ID: "call_2", Name: "t"}},
		litellm.BlockEnd{Index: 3},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v", events)
	}
	// The events satisfy the stream block lifecycle.
	stream := &sliceStream{events: append(events, litellm.DoneEvent{Provider: "test", Model: "m"})}
	if resp, err := litellm.Collect(stream); err != nil || len(resp.Blocks) != 4 {
		t.Fatalf("Collect: %v", err)
	}
}

type sliceStream struct{ events []litellm.Event }

func (s *sliceStream) Next() (litellm.Event, error) {
	if len(s.events) == 0 {
		return nil, io.EOF
	}
	event := s.events[0]
	s.events = s.events[1:]
	return event, nil
}

func (s *sliceStream) Close() error { return nil }
