package wire

import (
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/voocel/litellm"
)

func collect(t *testing.T, r *SSEReader) []SSEEvent {
	t.Helper()
	var events []SSEEvent
	for {
		event, err := r.Next()
		if errors.Is(err, io.EOF) {
			return events
		}
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
}

func TestReaderFraming(t *testing.T) {
	body := ": keepalive\r\nevent: delta\r\ndata: {\"a\":1}\r\n\r\ndata:{\"b\":2}\nid: 7\nretry: 10\ndata: [DONE]"
	got := collect(t, NewSSEReader(strings.NewReader(body), "test"))
	want := []SSEEvent{{Name: "delta", Data: `{"a":1}`}, {Data: `{"b":2}`}, {Data: "[DONE]"}}
	if len(got) != len(want) {
		t.Fatalf("events = %+v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestReaderEventNameAppliesToNextDataOnly(t *testing.T) {
	got := collect(t, NewSSEReader(strings.NewReader("event: a\n\ndata: x\n"), "test"))
	if len(got) != 1 || got[0].Name != "" {
		t.Fatalf("blank line must reset the event name: %+v", got)
	}
}

func TestReaderBareLines(t *testing.T) {
	r := NewSSEReader(strings.NewReader("[{\"a\":1},\n{\"b\":2}]\nid: 1\n"), "test")
	r.AcceptBare = true
	got := collect(t, r)
	if len(got) != 2 || got[0].Data != `[{"a":1},` || got[1].Data != `{"b":2}]` {
		t.Fatalf("events = %+v", got)
	}
	if got := collect(t, NewSSEReader(strings.NewReader("{\"a\":1}\n"), "test")); len(got) != 0 {
		t.Fatalf("bare lines must be ignored by default: %+v", got)
	}
}

func TestReaderLongLine(t *testing.T) {
	payload := strings.Repeat("x", 3<<20)
	got := collect(t, NewSSEReader(iotest.HalfReader(strings.NewReader("data: "+payload+"\n")), "test"))
	if len(got) != 1 || got[0].Data != payload {
		t.Fatal("long data line was not returned intact")
	}
}

func TestReaderReadFailureIsNetworkError(t *testing.T) {
	_, err := NewSSEReader(iotest.ErrReader(errors.New("reset")), "test").Next()
	if !litellm.IsNetworkError(err) {
		t.Fatalf("err = %v", err)
	}
}
