// Package provider builds the built-in providers by name, for applications
// that choose a provider from configuration. Its subpackages, such as
// provider/openai, are the providers themselves; settings beyond Config are
// set by building a provider from its own package.
package provider

import (
	"context"
	"fmt"
	"maps"
	"slices"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/anthropic"
	"github.com/voocel/litellm/provider/bedrock"
	"github.com/voocel/litellm/provider/compat"
	"github.com/voocel/litellm/provider/deepseek"
	"github.com/voocel/litellm/provider/gemini"
	"github.com/voocel/litellm/provider/glm"
	"github.com/voocel/litellm/provider/grok"
	"github.com/voocel/litellm/provider/mimo"
	"github.com/voocel/litellm/provider/minimax"
	"github.com/voocel/litellm/provider/ollama"
	"github.com/voocel/litellm/provider/openai"
	"github.com/voocel/litellm/provider/openrouter"
	"github.com/voocel/litellm/provider/qwen"
)

// Config holds the connection settings the providers share, plus the few
// that belong to one provider. Bedrock signs requests with Region and
// Credentials instead of an API key, headers or user agent.
type Config struct {
	// APIKey authenticates requests; APIKeyFunc, when set, resolves it per
	// request instead.
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// BaseURL overrides the provider's endpoint; compat requires it.
	BaseURL string
	// HTTPClient sends requests; nil uses http.DefaultClient. Wrap it with
	// retry.NewHTTPClient to retry.
	HTTPClient litellm.HTTPClient
	UserAgent  string
	// Headers are set after the defaults, so they can override them.
	Headers map[string]string

	// API selects the openai endpoint: openai.APIChat (the default) or
	// openai.APIResponses.
	API string
	// AllowUnknownProviderOptions makes the Chat Completions vendors
	// (deepseek, glm, grok, mimo, minimax, ollama, openrouter, qwen) copy
	// options they do not list into the request body, as compat always does.
	AllowUnknownProviderOptions bool
	// Region and Credentials configure bedrock.
	Region      string
	Credentials bedrock.CredentialsProvider
}

// New returns the provider called name, one of Names.
func New(name string, cfg Config) (litellm.Provider, error) {
	build, ok := builders[name]
	if !ok {
		return nil, litellm.NewError("", litellm.ErrorTypeValidation, fmt.Sprintf("unknown provider %q", name), nil)
	}
	return build(cfg)
}

// Names returns the provider names New accepts, sorted.
func Names() []string {
	return slices.Sorted(maps.Keys(builders))
}

var builders = map[string]func(Config) (litellm.Provider, error){
	"anthropic": func(c Config) (litellm.Provider, error) {
		return asProvider(anthropic.New(anthropic.Config{
			APIKey: c.APIKey, APIKeyFunc: c.APIKeyFunc, BaseURL: c.BaseURL,
			HTTPClient: c.HTTPClient, UserAgent: c.UserAgent, Headers: c.Headers,
		}))
	},
	"bedrock": func(c Config) (litellm.Provider, error) {
		return asProvider(bedrock.New(bedrock.Config{
			Region: c.Region, BaseURL: c.BaseURL, Credentials: c.Credentials, HTTPClient: c.HTTPClient,
		}))
	},
	"compat":   chatCompletions(compat.New),
	"deepseek": chatCompletions(deepseek.New),
	"gemini": func(c Config) (litellm.Provider, error) {
		return asProvider(gemini.New(gemini.Config{
			APIKey: c.APIKey, APIKeyFunc: c.APIKeyFunc, BaseURL: c.BaseURL,
			HTTPClient: c.HTTPClient, UserAgent: c.UserAgent, Headers: c.Headers,
		}))
	},
	"glm":     chatCompletions(glm.New),
	"grok":    chatCompletions(grok.New),
	"mimo":    chatCompletions(mimo.New),
	"minimax": chatCompletions(minimax.New),
	"ollama":  chatCompletions(ollama.New),
	"openai": func(c Config) (litellm.Provider, error) {
		return asProvider(openai.New(openai.Config{
			API: c.API, APIKey: c.APIKey, APIKeyFunc: c.APIKeyFunc, BaseURL: c.BaseURL,
			HTTPClient: c.HTTPClient, UserAgent: c.UserAgent, Headers: c.Headers,
		}))
	},
	"openrouter": chatCompletions(openrouter.New),
	"qwen":       chatCompletions(qwen.New),
}

// chatCompletions builds a Chat Completions vendor; they share compat.Config.
func chatCompletions[P litellm.Provider](build func(compat.Config) (P, error)) func(Config) (litellm.Provider, error) {
	return func(c Config) (litellm.Provider, error) {
		return asProvider(build(compat.Config{
			APIKey: c.APIKey, APIKeyFunc: c.APIKeyFunc, BaseURL: c.BaseURL,
			HTTPClient: c.HTTPClient, UserAgent: c.UserAgent, Headers: c.Headers,
			AllowUnknownProviderOptions: c.AllowUnknownProviderOptions,
		}))
	}
}

// asProvider returns a failed build as a nil Provider, not a typed nil.
func asProvider[P litellm.Provider](p P, err error) (litellm.Provider, error) {
	if err != nil {
		return nil, err
	}
	return p, nil
}
