// Package anthropic connects to the Anthropic Messages API.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Config configures the Messages API client. An API key is required.
type Config struct {
	// Name identifies the provider in responses, errors and ProviderState,
	// "anthropic" by default. Give each Messages-compatible endpoint its own
	// name, so thinking signatures are replayed only where they were issued.
	Name string
	// APIKey authenticates requests; APIKeyFunc, when set, resolves it per
	// request instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// BaseURL is the API origin, https://api.anthropic.com by default.
	BaseURL string
	// HTTPClient sends requests; nil uses http.DefaultClient. Wrap it with
	// retry.NewHTTPClient to retry.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are set after the defaults, so they can override them. Use them
	// for anthropic-beta, or to change anthropic-version from 2023-06-01.
	Headers map[string]string
}

// Provider implements litellm.Provider and litellm.CapabilityProvider.
type Provider struct {
	cfg Config
}

// New returns a Provider for cfg.
func New(cfg Config) (*Provider, error) {
	if cfg.Name == "" {
		cfg.Name = "anthropic"
	}
	if cfg.APIKey == "" && cfg.APIKeyFunc == nil {
		return nil, litellm.NewError(cfg.Name, litellm.ErrorTypeValidation, "api key is required", nil)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	cfg.HTTPClient = wire.HTTPClient(cfg.HTTPClient)
	if cfg.UserAgent == "" {
		cfg.UserAgent = wire.DefaultUserAgent
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the configured name.
func (p *Provider) Name() string {
	return p.cfg.Name
}

// Capabilities reports the static protocol facts.
func (p *Provider) Capabilities() litellm.Capabilities {
	return litellm.Capabilities{MaxTokensRequired: true, ThinkingEffort: true, DisableThinking: true, ProviderOptions: sortedOptions()}
}

// Chat sends a Messages request.
func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	resp, err := p.post(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, litellm.NewNetworkError(p.Name(), "read response failed", err)
	}
	var parsed response
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode response", err)
	}
	out := convertResponse(&parsed, p.Name(), req.Model)
	out.Raw = data
	return out, nil
}

// Stream sends a streaming Messages request.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	resp, err := p.post(ctx, req, true)
	if err != nil {
		return nil, err
	}
	return newStream(resp, p.Name(), req.Model), nil
}

func (p *Provider) post(ctx context.Context, req *litellm.Request, stream bool) (*http.Response, error) {
	body, err := buildRequest(req, p.Name(), stream)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.cfg.BaseURL, "/")+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	key, err := wire.APIKey(ctx, p.cfg.APIKey, p.cfg.APIKeyFunc, true)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", p.cfg.UserAgent)
	httpReq.Header.Set("x-api-key", key)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	if err := wire.SetHeaders(httpReq.Header, p.cfg.Headers); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	return wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "request")
}
