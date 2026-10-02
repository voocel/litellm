// Package openaicompat implements the Chat Completions protocol shared by
// OpenAI and the OpenAI-compatible providers. A Spec captures one vendor's
// dialect.
package openaicompat

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Provider speaks one Chat Completions dialect.
type Provider struct {
	cfg  Config
	spec Spec
}

// New returns a provider for spec; BaseURL falls back to spec.BaseURL.
func New(cfg Config, spec Spec) (*Provider, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = spec.BaseURL
	}
	if cfg.BaseURL == "" {
		return nil, litellm.NewError(spec.Name, litellm.ErrorTypeValidation, "base url is required", nil)
	}
	if spec.APIKeyRequired && cfg.APIKey == "" && cfg.APIKeyFunc == nil {
		return nil, litellm.NewError(spec.Name, litellm.ErrorTypeValidation, "api key is required", nil)
	}
	cfg.HTTPClient = wire.HTTPClient(cfg.HTTPClient)
	if cfg.UserAgent == "" {
		cfg.UserAgent = wire.DefaultUserAgent
	}
	return &Provider{cfg: cfg, spec: spec}, nil
}

// Name returns Config.Name, or else the spec name.
func (p *Provider) Name() string {
	if p.cfg.Name != "" {
		return p.cfg.Name
	}
	return p.spec.Name
}

// Capabilities lists the accepted provider options and the thinking settings
// the dialect's mapping takes.
func (p *Provider) Capabilities() litellm.Capabilities {
	options := slices.Clone(p.spec.Options)
	slices.Sort(options)
	return litellm.Capabilities{
		ThinkingEffort:  p.takes(litellm.Thinking{Effort: "high"}),
		DisableThinking: p.takes(litellm.Thinking{Disabled: true}),
		ProviderOptions: options,
	}
}

// takes reports whether the thinking mapping accepts thinking.
func (p *Provider) takes(thinking litellm.Thinking) bool {
	_, err := p.convertThinking(&thinking)
	return err == nil
}

// Chat sends a non-streaming Chat Completions request.
func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	body, err := p.buildRequest(req, false)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := p.post(ctx, body, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, litellm.NewNetworkError(p.Name(), "read response failed", err)
	}
	var parsed chatResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode response", err)
	}
	if err := wire.ErrorField(p.Name(), parsed.Error); err != nil {
		return nil, err
	}
	out, err := p.convertResponse(&parsed, req)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeProvider, err)
	}
	out.Raw = data
	if p.spec.usesSchemaPrompt(req.ResponseFormat) {
		out.Warnings = append(out.Warnings, p.spec.schemaWarning())
	}
	return out, nil
}

// Stream sends a streaming Chat Completions request.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	body, err := p.buildRequest(req, true)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := p.post(ctx, body, true)
	if err != nil {
		return nil, err
	}
	s := newStream(resp, req, p.Name(), p.spec)
	if p.spec.usesSchemaPrompt(req.ResponseFormat) {
		s.pending = append(s.pending, litellm.WarningEvent{Warning: p.spec.schemaWarning()})
	}
	return s, nil
}

func (p *Provider) post(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url("/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	if err := p.setHeaders(ctx, httpReq, stream); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	return wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "request")
}

func (p *Provider) setHeaders(ctx context.Context, httpReq *http.Request, stream bool) error {
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", p.cfg.UserAgent)
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	key, err := wire.APIKey(ctx, p.cfg.APIKey, p.cfg.APIKeyFunc, p.spec.APIKeyRequired)
	if err != nil {
		return err
	}
	if key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}
	return wire.SetHeaders(httpReq.Header, p.cfg.Headers)
}

func (p *Provider) url(path string) string {
	return strings.TrimRight(p.cfg.BaseURL, "/") + path
}
