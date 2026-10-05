package wire

import (
	"io"

	"github.com/voocel/litellm"
)

// Stream is the litellm.Stream of a vendor response body. Next delivers the
// queued events and, while none are queued, calls read, which decodes the
// next frame of the body and appends its events to those it is given. read
// returns io.EOF at the end of the body, which before a DoneEvent the Client
// reports as a truncated reply, a retryable network error. Once read queues
// a DoneEvent or fails, or the stream is closed, Next returns io.EOF after
// the queued events.
type Stream struct {
	body    io.Closer
	read    func([]litellm.Event) ([]litellm.Event, error)
	pending []litellm.Event
	done    bool
}

// NewStream returns a Stream over body that delivers events first.
func NewStream(body io.Closer, read func([]litellm.Event) ([]litellm.Event, error), events ...litellm.Event) *Stream {
	return &Stream{body: body, read: read, pending: events}
}

// Next returns the next event.
func (s *Stream) Next() (litellm.Event, error) {
	for len(s.pending) == 0 {
		if s.done {
			return nil, io.EOF
		}
		events, err := s.read(s.pending)
		if err != nil {
			s.done = true
			return nil, err
		}
		s.pending = events
		// A DoneEvent is always the last event of its frame.
		if n := len(events); n > 0 {
			_, s.done = events[n-1].(litellm.DoneEvent)
		}
	}
	event := s.pending[0]
	s.pending = s.pending[1:]
	return event, nil
}

// Close closes the body.
func (s *Stream) Close() error {
	s.done, s.pending = true, nil
	return s.body.Close()
}
