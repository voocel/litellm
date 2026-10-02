package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
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
	// or to cap its tokens. An error refuses the call, with the type and
	// status a litellm.Error carries or else as a model not found (404).
	Route func(r *http.Request, req *litellm.Request) (*litellm.Client, error)
}

// ServeHTTP makes the call r carries, once: retrying is the caller's.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, &wireError{Type: litellm.ErrorTypeValidation, Message: "method not allowed", Provider: "gateway"})
		return
	}
	var req litellm.Request
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, MaxRequestBytes)).Decode(&req); err != nil {
		code := http.StatusBadRequest
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			code = http.StatusRequestEntityTooLarge
		}
		writeError(w, code, &wireError{Type: litellm.ErrorTypeValidation, Message: fmt.Sprintf("bad request: %v", err), Provider: "gateway"})
		return
	}
	client, err := s.Route(r, &req)
	if err != nil {
		writeError(w, status(err, http.StatusNotFound), toWireError(err, litellm.ErrorTypeModel))
		return
	}
	stream, err := client.Stream(r.Context(), req)
	if err != nil {
		err = upstream(err)
		writeError(w, status(err, http.StatusBadGateway), toWireError(err, litellm.ErrorTypeProvider))
		return
	}
	defer stream.Close()

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("X-Accel-Buffering", "no") // keep proxies from holding the stream back
	reply := newReply(w)
	defer reply.close()
	// A failed send means the caller is gone; closing the stream ends the
	// upstream call.
	var sendErr error
	_, err = litellm.Handle(stream, func(ev litellm.Event) error {
		sendErr = reply.send(toEvent(ev))
		return sendErr
	})
	if err != nil && sendErr == nil {
		reply.send(event{Type: "error", Error: toWireError(upstream(err), litellm.ErrorTypeProvider)})
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

// reply writes the lines of a reply, with a heartbeat after each
// heartbeatInterval without one.
type reply struct {
	mu     sync.Mutex
	enc    *json.Encoder
	flush  func() error
	timer  *time.Timer
	closed bool
}

func newReply(w http.ResponseWriter) *reply {
	r := &reply{enc: json.NewEncoder(w), flush: http.NewResponseController(w).Flush}
	r.mu.Lock() // a heartbeat waits for the timer to be set
	defer r.mu.Unlock()
	r.timer = time.AfterFunc(heartbeatInterval, r.heartbeat)
	return r
}

func (r *reply) send(e event) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.timer.Reset(heartbeatInterval)
	if err := r.enc.Encode(e); err != nil {
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
	if r.enc.Encode(event{Type: heartbeat}) == nil {
		r.flush()
	}
	r.timer.Reset(heartbeatInterval)
}

// close stops the heartbeats; the ResponseWriter is not used after it.
func (r *reply) close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	r.timer.Stop()
}

// status is the HTTP status of a call refused with err: the upstream status
// a litellm.Error carries, or fallback.
func status(err error, fallback int) int {
	var e *litellm.Error
	if errors.As(err, &e) && e.StatusCode >= 400 {
		return e.StatusCode
	}
	return fallback
}

func writeError(w http.ResponseWriter, status int, e *wireError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(errorBody{e})
}

// errorBody is the body of a refused call.
type errorBody struct {
	Error *wireError `json:"error"`
}
