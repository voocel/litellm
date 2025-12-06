package litellm

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// Router interface defines how to select a provider for a given model
type Router interface {
	Route(model string, availableProviders []Provider) (Provider, error)
}

// RouteStrategy defines different routing strategies
type RouteStrategy string

const (
	StrategyAuto       RouteStrategy = "auto"        // Intelligent routing based on context
	StrategyExact      RouteStrategy = "exact"       // Exact model match required
	StrategyFirst      RouteStrategy = "first"       // Use first available provider
	StrategyRoundRobin RouteStrategy = "round_robin" // Round-robin selection
)

// maxCacheSize limits the router cache to prevent unbounded memory growth
const maxCacheSize = 1000

// SmartRouter implements intelligent routing logic
type SmartRouter struct {
	strategy         RouteStrategy
	fallbackStrategy FallbackStrategy
	roundRobinIndex  atomic.Int64 // For round-robin strategy (thread-safe)
	cache            sync.Map     // Cache for auto routing results
	cacheSize        atomic.Int64 // Track cache size to prevent unbounded growth
}

// FallbackStrategy defines what to do when primary routing fails
type FallbackStrategy string

const (
	FallbackNone  FallbackStrategy = "none"  // Fail immediately
	FallbackFirst FallbackStrategy = "first" // Use first available provider
	FallbackAny   FallbackStrategy = "any"   // Try any provider that supports the capability
	FallbackBest  FallbackStrategy = "best"  // Use provider with best capability match
)

// NewSmartRouter creates a new smart router
func NewSmartRouter(strategy RouteStrategy) *SmartRouter {
	return &SmartRouter{
		strategy:         strategy,
		fallbackStrategy: FallbackFirst,
	}
}

// WithFallback sets the fallback strategy
func (r *SmartRouter) WithFallback(fallback FallbackStrategy) *SmartRouter {
	r.fallbackStrategy = fallback
	return r
}

// Route implements the Router interface
func (r *SmartRouter) Route(model string, availableProviders []Provider) (Provider, error) {
	if len(availableProviders) == 0 {
		return nil, NewError(ErrorTypeValidation, "no providers available")
	}

	// Single provider scenario - simplified routing
	if len(availableProviders) == 1 {
		provider := availableProviders[0]
		// For single provider, we accept any model name and let the API validate
		// This follows the "progressive complexity" principle
		return provider, nil
	}

	// Multiple providers - need smart routing
	switch r.strategy {
	case StrategyExact:
		return r.routeExact(model, availableProviders)
	case StrategyFirst:
		return r.routeFirst(model, availableProviders)
	case StrategyRoundRobin:
		return r.routeRoundRobin(model, availableProviders)
	case StrategyAuto:
		return r.routeAuto(model, availableProviders)
	default:
		return r.routeAuto(model, availableProviders)
	}
}

// routeExact finds exact model match
func (r *SmartRouter) routeExact(model string, providers []Provider) (Provider, error) {
	for _, provider := range providers {
		if provider.SupportsModel(model) {
			return provider, nil
		}
	}

	return r.applyFallback(model, providers, fmt.Errorf("no provider supports model '%s'", model))
}

// routeFirst returns first provider that supports the model
func (r *SmartRouter) routeFirst(model string, providers []Provider) (Provider, error) {
	for _, provider := range providers {
		if provider.SupportsModel(model) {
			return provider, nil
		}
	}

	return r.applyFallback(model, providers, fmt.Errorf("no provider supports model '%s'", model))
}

// routeRoundRobin implements round-robin selection among supporting providers
func (r *SmartRouter) routeRoundRobin(model string, providers []Provider) (Provider, error) {
	// Find all providers that support this model
	var supportingProviders []Provider
	for _, provider := range providers {
		if provider.SupportsModel(model) {
			supportingProviders = append(supportingProviders, provider)
		}
	}

	if len(supportingProviders) == 0 {
		return r.applyFallback(model, providers, fmt.Errorf("no provider supports model '%s'", model))
	}

	// Select using round-robin with atomic counter for thread-safety
	index := r.roundRobinIndex.Add(1) - 1
	selectedProvider := supportingProviders[index%int64(len(supportingProviders))]

	return selectedProvider, nil
}

// routeAuto implements intelligent automatic routing
func (r *SmartRouter) routeAuto(model string, providers []Provider) (Provider, error) {
	// Check cache first
	if cached, ok := r.cache.Load(model); ok {
		return cached.(Provider), nil
	}

	// Helper to cache and return (with size limit to prevent OOM)
	returnWithCache := func(p Provider) (Provider, error) {
		// Only cache if below size limit
		if r.cacheSize.Load() < maxCacheSize {
			r.cache.Store(model, p)
			r.cacheSize.Add(1)
		}
		return p, nil
	}

	// Strategy 1: Try exact model match first
	for _, provider := range providers {
		if provider.SupportsModel(model) {
			return returnWithCache(provider)
		}
	}

	// Strategy 2: Fuzzy matching for common model patterns
	if provider := r.fuzzyMatch(model, providers); provider != nil {
		return returnWithCache(provider)
	}

	// Strategy 3: Provider inference from model name patterns
	if provider := r.inferProviderFromModel(model, providers); provider != nil {
		return returnWithCache(provider)
	}

	return r.applyFallback(model, providers, fmt.Errorf("no suitable provider found for model '%s'", model))
}

// fuzzyMatch attempts to match model names with some flexibility
func (r *SmartRouter) fuzzyMatch(model string, providers []Provider) Provider {
	modelLower := strings.ToLower(model)

	for _, provider := range providers {
		models := provider.Models()
		for _, modelInfo := range models {
			modelIDLower := strings.ToLower(modelInfo.ID)

			// Check for partial matches or aliases
			if strings.Contains(modelIDLower, modelLower) || strings.Contains(modelLower, modelIDLower) {
				return provider
			}

			// Check for common aliases/variations
			if r.isModelAlias(modelLower, modelIDLower) {
				return provider
			}
		}
	}

	return nil
}

// inferProviderFromModel infers provider from model name patterns
func (r *SmartRouter) inferProviderFromModel(model string, providers []Provider) Provider {
	modelLower := strings.ToLower(model)

	// Common model name patterns
	patterns := map[string][]string{
		"openai":    {"gpt", "o1", "o3", "davinci", "curie", "babbage", "ada"},
		"anthropic": {"claude"},
		"gemini":    {"gemini", "bard"},
		"deepseek":  {"deepseek"},
		"bedrock":   {"anthropic.", "amazon.", "meta.", "mistral.", "cohere.", "ai21."},
	}

	for _, provider := range providers {
		providerName := strings.ToLower(provider.Name())
		if keywords, exists := patterns[providerName]; exists {
			for _, keyword := range keywords {
				if strings.Contains(modelLower, keyword) {
					return provider
				}
			}
		}
	}

	return nil
}

// isModelAlias checks for known model aliases
func (r *SmartRouter) isModelAlias(model1, model2 string) bool {
	aliases := map[string][]string{
		"gpt-4":    {"gpt4", "gpt-4.0"},
		"gpt-4o":   {"gpt4o"},
		"claude-4": {"claude4", "claude-4.0"},
	}

	for canonical, aliasList := range aliases {
		for _, alias := range aliasList {
			if (model1 == canonical && model2 == alias) || (model1 == alias && model2 == canonical) {
				return true
			}
		}
	}

	return false
}

// applyFallback applies the configured fallback strategy
func (r *SmartRouter) applyFallback(model string, providers []Provider, originalErr error) (Provider, error) {
	switch r.fallbackStrategy {
	case FallbackNone:
		return nil, NewModelError("router", model, originalErr.Error())

	case FallbackFirst:
		if len(providers) > 0 {
			return providers[0], nil
		}
		return nil, NewError(ErrorTypeValidation, "no providers available for fallback")

	case FallbackAny:
		// Return any provider (first available)
		if len(providers) > 0 {
			return providers[0], nil
		}
		return nil, NewError(ErrorTypeValidation, "no providers available for fallback")

	case FallbackBest:
		return r.selectBestProvider(model, providers)

	default:
		return nil, NewModelError("router", model, originalErr.Error())
	}
}

// selectBestProvider selects the provider with the best capability match
func (r *SmartRouter) selectBestProvider(model string, providers []Provider) (Provider, error) {
	if len(providers) == 0 {
		return nil, NewError(ErrorTypeValidation, "no providers available")
	}

	// Simple heuristic: prefer provider with more models (more likely to support similar models)
	var bestProvider Provider
	maxModels := 0

	for _, provider := range providers {
		models := provider.Models()
		if len(models) > maxModels {
			maxModels = len(models)
			bestProvider = provider
		}
	}

	if bestProvider != nil {
		return bestProvider, nil
	}

	// Fallback to first provider
	return providers[0], nil
}

// CustomRouterFunc allows users to provide custom routing logic
type CustomRouterFunc func(model string, providers []Provider) (Provider, error)

// Route implements Router interface for CustomRouterFunc
func (f CustomRouterFunc) Route(model string, providers []Provider) (Provider, error) {
	return f(model, providers)
}

// Common router constructors

// NewExactRouter creates a router that requires exact model matches
func NewExactRouter() *SmartRouter {
	return NewSmartRouter(StrategyExact)
}

// NewFirstRouter creates a router that uses the first matching provider
func NewFirstRouter() *SmartRouter {
	return NewSmartRouter(StrategyFirst)
}

// NewRoundRobinRouter creates a router that distributes requests round-robin
func NewRoundRobinRouter() *SmartRouter {
	return NewSmartRouter(StrategyRoundRobin)
}

// NewAutoRouter creates a router with intelligent automatic routing
func NewAutoRouter() *SmartRouter {
	return NewSmartRouter(StrategyAuto)
}

// Default router instance (By default, do not downgrade to avoid misrouting)
var DefaultRouter = NewAutoRouter().WithFallback(FallbackNone)

// Convenience functions for common routing needs

// RouteToProvider creates a simple router that always returns a specific provider
func RouteToProvider(provider Provider) Router {
	return CustomRouterFunc(func(model string, providers []Provider) (Provider, error) {
		for _, p := range providers {
			if p.Name() == provider.Name() {
				return provider, nil
			}
		}
		return nil, NewError(ErrorTypeValidation, "specified provider not available")
	})
}

// RouteByProviderName creates a router that selects provider by name
func RouteByProviderName(providerName string) Router {
	return CustomRouterFunc(func(model string, providers []Provider) (Provider, error) {
		for _, provider := range providers {
			if strings.EqualFold(provider.Name(), providerName) {
				return provider, nil
			}
		}
		return nil, NewError(ErrorTypeValidation, fmt.Sprintf("provider '%s' not found", providerName))
	})
}
