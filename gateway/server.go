package gateway

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/voocel/litellm"
)

// MaxRequestBytes caps the body of a call; a larger one is refused with 413.
const MaxRequestBytes = 64 << 20

// heartbeatInterval is the upstream silence after which the Server sends a
// heartbeat line, well inside the idle timeouts of common proxies.
var heartbeatInterval = 15 * time.Second

// Server serves model calls. It is an http.Handler.
type Server struct {
	// Route returns the Client to make the call req for r with. It may
	// rewrite req, such as to map the model the caller names to the vendor's,
	// or to cap its tokens. An error refuses the call: a litellm.Error with
	// its type, any other as a model not found.
	Route func(r *http.Request, req *litellm.Request) (*litellm.Client, error)
}

// ServeHTTP makes the call r carries, once: retrying is the caller's.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, &wireError{Type: litellm.ErrorTypeValidation, StatusCode: http.StatusMethodNotAllowed, Message: "method not allowed", Provider: "gateway"})
		return
	}
	if !canFlush(w) {
		writeError(w, &wireError{Type: litellm.ErrorTypeInternal, Provider: "gateway",
			Message: "the ResponseWriter cannot flush, so a reply cannot stream: a middleware's ResponseWriter must implement http.Flusher or Unwrap"})
		return
	}
	var req litellm.Request
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, MaxRequestBytes)).Decode(&req); err != nil {
		e := &wireError{Type: litellm.ErrorTypeValidation, Message: fmt.Sprintf("bad request: %v", err), Provider: "gateway"}
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			e.StatusCode = http.StatusRequestEntityTooLarge
		}
		writeError(w, e)
		return
	}
	client, err := s.Route(r, &req)
	if err != nil {
		writeError(w, toWireError(err, litellm.ErrorTypeModel))
		return
	}
	// Heartbeats start before the upstream answers, which a retry waiting
	// out a Retry-After may delay.
	reply := newReply(w)
	defer reply.close()
	stream, err := client.Stream(r.Context(), req)
	if err != nil {
		reply.fail(toWireError(upstream(err), litellm.ErrorTypeProvider))
		return
	}
	defer stream.Close()
	// A failed send means the caller is gone; closing the stream ends the
	// upstream call. Any other failure, an event that does not encode
	// included, goes to the caller.
	var sendErr error
	_, err = litellm.Handle(stream, func(ev litellm.Event) error {
		line, err := json.Marshal(toEvent(ev))
		if err != nil {
			return err
		}
		sendErr = reply.send(line)
		return sendErr
	})
	if err != nil && sendErr == nil {
		reply.fail(toWireError(upstream(err), litellm.ErrorTypeProvider))
	}
}

// canFlush reports whether w, or a ResponseWriter it unwraps to, can flush,
// as http.ResponseController looks for it.
func canFlush(w http.ResponseWriter) bool {
	for {
		switch t := w.(type) {
		case http.Flusher, interface{ FlushError() error }:
			return true
		case interface{ Unwrap() http.ResponseWriter }:
			w = t.Unwrap()
		default:
			return false
		}
	}
}

// upstream is the error of a failed call as the caller should see it. The
// upstream rejecting the Server's key is the Server's fault, not the
// caller's, so it is a provider error.
func upstream(err error) error {
	e, ok := errors.AsType[*litellm.Error](err)
	if !ok || e.Type != litellm.ErrorTypeAuth {
		return err
	}
	return &litellm.Error{Type: litellm.ErrorTypeProvider, Code: e.Code, Message: "upstream key rejected: " + e.Message, Provider: e.Provider}
}

// reply writes the reply to a call: a refusal, or lines of events with a
// heartbeat after each heartbeatInterval without one. The first line
// commits the reply to 200.
type reply struct {
	mu     sync.Mutex
	w      http.ResponseWriter
	flush  func() error
	timer  *time.Timer
	lines  bool
	closed bool
}

func newReply(w http.ResponseWriter) *reply {
	r := &reply{w: w, flush: http.NewResponseController(w).Flush}
	r.mu.Lock() // a heartbeat waits for the timer to be set
	defer r.mu.Unlock()
	r.timer = time.AfterFunc(heartbeatInterval, r.heartbeat)
	return r
}

// send writes line, an encoded event, and flushes it.
func (r *reply) send(line []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.timer.Reset(heartbeatInterval)
	return r.write(line)
}

// fail ends the reply with e: a refusal while no line has gone out, else an
// error event.
func (r *reply) fail(e *wireError) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	r.timer.Stop()
	if !r.lines {
		writeError(r.w, e)
		return
	}
	line, _ := json.Marshal(event{Type: "error", Error: e}) // an error event always encodes
	r.write(line)
}

func (r *reply) write(line []byte) error {
	if !r.lines {
		r.lines = true
		r.w.Header().Set("Content-Type", "application/x-ndjson")
		r.w.Header().Set("X-Accel-Buffering", "no") // keep proxies from holding the stream back
	}
	if _, err := r.w.Write(append(line, '\n')); err != nil {
		return err
	}
	return r.flush()
}

// heartbeat ignores a failed write: the next send reports it.
func (r *reply) heartbeat() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return
	}
	r.write([]byte(`{"type":"` + heartbeat + `"}`))
	r.timer.Reset(heartbeatInterval)
}

// close stops the heartbeats; the ResponseWriter is not used after it.
func (r *reply) close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	r.timer.Stop()
}

// writeError refuses a call with e, with the status of the upstream response
// it reports or else the one its type has, and the wait it suggests.
func writeError(w http.ResponseWriter, e *wireError) {
	e.StatusCode = cmp.Or(e.StatusCode, statusOf(e.Type))
	w.Header().Set("Content-Type", "application/json")
	if e.RetryAfterMS > 0 {
		w.Header().Set("Retry-After", strconv.FormatInt((e.RetryAfterMS+999)/1000, 10))
	}
	w.WriteHeader(e.StatusCode)
	json.NewEncoder(w).Encode(errorBody{e})
}

// statusOf is the status vendors refuse a call of type t with; a failure of
// the upstream itself is a bad gateway.
func statusOf(t litellm.ErrorType) int {
	switch t {
	case litellm.ErrorTypeValidation, litellm.ErrorTypeContextOverflow, litellm.ErrorTypeContentFilter:
		return http.StatusBadRequest
	case litellm.ErrorTypeAuth:
		return http.StatusUnauthorized
	case litellm.ErrorTypeQuota:
		return http.StatusPaymentRequired
	case litellm.ErrorTypeModel:
		return http.StatusNotFound
	case litellm.ErrorTypeRateLimit:
		return http.StatusTooManyRequests
	case litellm.ErrorTypeInternal:
		return http.StatusInternalServerError
	case litellm.ErrorTypeOverloaded:
		return http.StatusServiceUnavailable
	case litellm.ErrorTypeTimeout:
		return http.StatusGatewayTimeout
	default:
		return http.StatusBadGateway
	}
}

// errorBody is the body of a refused call.
type errorBody struct {
	Error *wireError `json:"error"`
}
