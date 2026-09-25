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

// Name returns the spec name.
func (p *Provider) Name() string {
	return p.spec.Name
}

// Capabilities probes the thinking mapping, which is the single source of
// truth for what the dialect can express.
func (p *Provider) Capabilities() litellm.Capabilities {
	accepts := func(t litellm.Thinking) bool {
		_, err := p.convertThinking(&t)
		return err == nil
	}
	options := slices.Clone(p.spec.Options)
	slices.Sort(options)
	return litellm.Capabilities{
		Thinking:        true,
		DisableThinking: accepts(litellm.Thinking{Mode: litellm.ThinkingDisabled}),
		ThinkingEffort:  accepts(litellm.Thinking{Effort: "high"}),
		ThinkingBudget:  accepts(litellm.Thinking{BudgetTokens: new(1024)}),
		ProviderOptions: options,
	}
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
	out, err := p.convertResponse(&parsed, req)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeProvider, err)
	}
	out.Raw = data
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
	return newStream(resp, req, p.spec), nil
}

// ListModels calls GET /models.
func (p *Provider) ListModels(ctx context.Context) ([]litellm.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, p.url("/models"), nil)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create models request", err)
	}
	if err := p.setHeaders(ctx, httpReq, false); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	resp, err := wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "models request")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var payload modelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode models response", err)
	}
	models := make([]litellm.ModelInfo, 0, len(payload.Data))
	for _, item := range payload.Data {
		name := item.Name
		if name == "" {
			name = item.ID
		}
		models = append(models, litellm.ModelInfo{
			ID:            item.ID,
			Name:          name,
			Provider:      p.Name(),
			Description:   item.Description,
			Created:       item.Created,
			ContextLength: item.ContextLength,
		})
	}
	return models, nil
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
