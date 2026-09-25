// Package openai connects to the OpenAI API, or any endpoint that implements
// it exactly (such as Azure OpenAI), through Chat Completions or Responses.
// Use provider/compat for other OpenAI-compatible servers.
package openai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
	"github.com/voocel/litellm/provider/internal/openaicompat"
)

// Config.API values.
const (
	APIChat      = "chat"
	APIResponses = "responses"
)

// Config configures the OpenAI client. An API key is required.
type Config struct {
	// API selects Chat Completions (APIChat, the default) or APIResponses.
	API string
	// APIKey authenticates requests; APIKeyFunc, when set, resolves it per
	// request instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// BaseURL is the API root, https://api.openai.com/v1 by default.
	BaseURL string
	// HTTPClient sends requests; nil uses http.DefaultClient. Wrap it with
	// retry.NewHTTPClient to retry.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are set after the defaults, so they can override them.
	Headers map[string]string
}

// Provider implements litellm.Provider, litellm.CapabilityProvider and
// litellm.ModelLister.
type Provider struct {
	cfg  Config
	chat *openaicompat.Provider
}

// New returns a Provider for cfg.
func New(cfg Config) (*Provider, error) {
	switch cfg.API {
	case "":
		cfg.API = APIChat
	case APIChat, APIResponses:
	default:
		return nil, litellm.NewError("openai", litellm.ErrorTypeValidation, fmt.Sprintf("api must be %q or %q, got %q", APIChat, APIResponses, cfg.API), nil)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	cfg.HTTPClient = wire.HTTPClient(cfg.HTTPClient)
	if cfg.UserAgent == "" {
		cfg.UserAgent = wire.DefaultUserAgent
	}
	chat, err := openaicompat.New(openaicompat.Config{
		APIKey: cfg.APIKey, APIKeyFunc: cfg.APIKeyFunc, BaseURL: cfg.BaseURL,
		HTTPClient: cfg.HTTPClient, UserAgent: cfg.UserAgent, Headers: cfg.Headers,
	}, openaicompat.Spec{
		Name:           "openai",
		APIKeyRequired: true,
		MaxTokensField: "max_completion_tokens",
		Options:        chatOptions,
		Cache:          promptCacheBreakpoint,
	})
	if err != nil {
		return nil, err
	}
	return &Provider{cfg: cfg, chat: chat}, nil
}

// Name returns "openai".
func (p *Provider) Name() string {
	return "openai"
}

// Capabilities reports the static protocol facts of the selected API.
func (p *Provider) Capabilities() litellm.Capabilities {
	if p.cfg.API == APIResponses {
		return litellm.Capabilities{Thinking: true, DisableThinking: true, ThinkingEffort: true, ProviderOptions: sortedCopy(responsesOptions)}
	}
	return p.chat.Capabilities()
}

// Chat sends the request through the selected API.
func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	if err := p.checkOptions(req); err != nil {
		return nil, err
	}
	if p.cfg.API == APIResponses {
		return p.responses(ctx, req)
	}
	return p.chat.Chat(ctx, req)
}

// Stream sends a streaming request through the selected API.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	if err := p.checkOptions(req); err != nil {
		return nil, err
	}
	if p.cfg.API == APIResponses {
		return p.responsesStream(ctx, req)
	}
	return p.chat.Stream(ctx, req)
}

// ListModels calls GET /models.
func (p *Provider) ListModels(ctx context.Context) ([]litellm.ModelInfo, error) {
	return p.chat.ListModels(ctx)
}

// checkOptions points options of the other API to the Config.API that accepts
// them.
func (p *Provider) checkOptions(req *litellm.Request) error {
	own, other, name := chatOptions, responsesOptions, APIResponses
	if p.cfg.API == APIResponses {
		own, other, name = responsesOptions, chatOptions, APIChat
	}
	for key := range req.ProviderOptions {
		if !slices.Contains(own, key) && slices.Contains(other, key) {
			return litellm.NewError(p.Name(), litellm.ErrorTypeValidation, fmt.Sprintf("provider option %q requires Config.API %q", key, name), nil)
		}
	}
	return nil
}

// post sends a Responses request and returns the successful response.
func (p *Provider) post(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.cfg.BaseURL, "/")+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	key, err := wire.APIKey(ctx, p.cfg.APIKey, p.cfg.APIKeyFunc, true)
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key)
	httpReq.Header.Set("User-Agent", p.cfg.UserAgent)
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	if err := wire.SetHeaders(httpReq.Header, p.cfg.Headers); err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	return wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "request")
}

// promptCacheBreakpoint marks an explicit prompt cache breakpoint. OpenAI sets
// the TTL for the whole request through prompt_cache_options.
func promptCacheBreakpoint(cache *litellm.CacheControl) (map[string]any, error) {
	if cache.TTL != "" {
		return nil, errors.New("cache breakpoint TTL is set through the prompt_cache_options provider option")
	}
	return map[string]any{"prompt_cache_breakpoint": map[string]any{"mode": "explicit"}}, nil
}

func (p *Provider) readResponse(resp *http.Response) ([]byte, error) {
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, litellm.NewNetworkError(p.Name(), "read response failed", err)
	}
	return data, nil
}
