package gateway

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Config configures a Provider.
type Config struct {
	// Name names the Provider in its own errors, "gateway" by default.
	// Responses name the provider that made the call.
	Name string
	// BaseURL is the URL the Server is served at.
	BaseURL string
	// APIKey, if set, is sent as a bearer token; APIKeyFunc, when set,
	// resolves it per call instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// HTTPClient sends calls; nil uses http.DefaultClient.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are sent with every call, after the defaults.
	Headers map[string]string
}

// Provider is a litellm.Provider whose calls run on a Server.
type Provider struct {
	cfg Config
}

// New returns a Provider for the Server at cfg.BaseURL.
func New(cfg Config) (*Provider, error) {
	if cfg.Name == "" {
		cfg.Name = "gateway"
	}
	if cfg.BaseURL == "" {
		return nil, litellm.NewError(cfg.Name, litellm.ErrorTypeValidation, "base URL is required", nil)
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the configured name. Responses name the provider that made
// the call.
func (p *Provider) Name() string { return p.cfg.Name }

// Chat makes the call and collects its reply.
func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	stream, err := p.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	return litellm.Collect(stream)
}

// Stream makes the call and streams its reply.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeValidation, fmt.Sprintf("encode request: %v", err), nil)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.cfg.BaseURL, bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeValidation, err.Error(), nil)
	}
	key, err := wire.APIKey(ctx, p.cfg.APIKey, p.cfg.APIKeyFunc, false)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeAuth, err.Error(), err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/x-ndjson")
	httpReq.Header.Set("User-Agent", cmp.Or(p.cfg.UserAgent, wire.DefaultUserAgent))
	if key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}
	if err := wire.SetHeaders(httpReq.Header, p.cfg.Headers); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeValidation, err.Error(), nil)
	}

	resp, err := wire.HTTPClient(p.cfg.HTTPClient).Do(httpReq)
	if err != nil {
		return nil, litellm.NewNetworkError(p.Name(), "call failed", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, refused(p.Name(), resp)
	}
	return &stream{name: p.Name(), body: resp.Body, lines: bufio.NewReader(resp.Body)}, nil
}

// refused is the error of a call the Server, or something in front of it,
// refused. The Server explains itself in the body with an error of a
// litellm.ErrorType; anything else is judged as any vendor's reply would be.
func refused(name string, resp *http.Response) error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, wire.MaxErrorBody))
	var body errorBody
	if json.Unmarshal(data, &body) == nil && body.Error != nil && isErrorType(body.Error.Type) {
		return body.Error.toError()
	}
	return wire.HTTPError(name, resp.StatusCode, resp.Header, string(data))
}

type stream struct {
	name  string
	body  io.ReadCloser
	lines *bufio.Reader
	done  bool
}

// Next returns the event of the next line. A read that fails is a network
// error and a line that is not an event a bad reply. A reply that ends
// before done, within a line or not, is cut short, as Collect reports it.
func (s *stream) Next() (litellm.Event, error) {
	if s.done {
		return nil, io.EOF
	}
	var e event
	for {
		line, err := s.lines.ReadBytes('\n')
		if errors.Is(err, io.EOF) {
			s.done = true
			return nil, io.EOF
		}
		if err != nil {
			s.done = true
			return nil, litellm.NewNetworkError(s.name, "read reply", err)
		}
		e = event{}
		if err := json.Unmarshal(line, &e); err != nil {
			s.done = true
			return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "bad reply: "+err.Error(), nil)
		}
		if e.Type != heartbeat {
			break
		}
	}
	ev, err := fromEvent(e)
	if err != nil {
		s.done = true
		var le *litellm.Error
		if errors.As(err, &le) {
			return nil, le
		}
		return nil, litellm.NewError(s.name, litellm.ErrorTypeProvider, "bad reply: "+err.Error(), nil)
	}
	if _, ok := ev.(litellm.DoneEvent); ok {
		s.done = true
	}
	return ev, nil
}

func (s *stream) Close() error {
	s.done = true
	return s.body.Close()
}
