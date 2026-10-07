// Package gemini connects to the Gemini API.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Config configures the Gemini API client. An API key is required.
type Config struct {
	// Name, when set, names the provider instead of "gemini", as a
	// compatible endpoint should: provider state and errors carry it.
	Name string
	// APIKey authenticates requests; APIKeyFunc, when set, resolves it per
	// request instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// BaseURL is the API origin, https://generativelanguage.googleapis.com by
	// default.
	BaseURL string
	// HTTPClient sends requests; nil uses http.DefaultClient. Wrap it with
	// retry.NewHTTPClient to retry.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are set after the defaults, so they can override them.
	Headers map[string]string
}

// Provider implements litellm.Provider and litellm.CapabilityProvider.
type Provider struct {
	cfg Config
}

// New returns a Provider for cfg.
func New(cfg Config) (*Provider, error) {
	if cfg.Name == "" {
		cfg.Name = "gemini"
	}
	if cfg.APIKey == "" && cfg.APIKeyFunc == nil {
		return nil, litellm.NewError(cfg.Name, litellm.ErrorTypeValidation, "api key is required", nil)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://generativelanguage.googleapis.com"
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
	return litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: slices.Clone(providerOptions)}
}

// Chat sends a generateContent request.
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

// Stream sends a streamGenerateContent request.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	resp, err := p.post(ctx, req, true)
	if err != nil {
		return nil, err
	}
	return newStream(resp, p.Name(), req.Model), nil
}

func (p *Provider) post(ctx context.Context, req *litellm.Request, stream bool) (*http.Response, error) {
	body, err := buildRequest(req, p.Name())
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	method := "generateContent"
	if stream {
		method = "streamGenerateContent?alt=sse"
	}
	url := fmt.Sprintf("%s/v1beta/models/%s:%s", strings.TrimRight(p.cfg.BaseURL, "/"), req.Model, method)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	if err := p.setHeaders(ctx, httpReq); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	return wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "request")
}

// ListModels lists the models of GET /v1beta/models that generate content.
// The vendor has far fewer than the 1000 one page holds.
func (p *Provider) ListModels(ctx context.Context) ([]litellm.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.cfg.BaseURL, "/")+"/v1beta/models?pageSize=1000", nil)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	if err := p.setHeaders(ctx, httpReq); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "list models")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var list struct {
		Models []struct {
			Name        string   `json:"name"`
			DisplayName string   `json:"displayName"`
			Methods     []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode model list", err)
	}
	var out []litellm.ModelInfo
	for _, m := range list.Models {
		if slices.Contains(m.Methods, "generateContent") {
			out = append(out, litellm.ModelInfo{ID: strings.TrimPrefix(m.Name, "models/"), Name: m.DisplayName})
		}
	}
	return out, nil
}

func (p *Provider) setHeaders(ctx context.Context, req *http.Request) error {
	key, err := wire.APIKey(ctx, p.cfg.APIKey, p.cfg.APIKeyFunc, true)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", p.cfg.UserAgent)
	req.Header.Set("x-goog-api-key", key)
	return wire.SetHeaders(req.Header, p.cfg.Headers)
}
