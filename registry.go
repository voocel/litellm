package litellm

import (
	"fmt"
	"strings"
	"sync"

	"github.com/voocel/litellm/providers"
)

// Global registry for custom providers
var (
	customProviders = make(map[string]ProviderFactory)
	providerMutex   sync.RWMutex
)

func normalizeProviderName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

// createProvider creates a provider instance by name
func createProvider(name string, config ProviderConfig) (Provider, error) {
	providerName := normalizeProviderName(name)

	// Check custom providers first
	providerMutex.RLock()
	if factory, exists := customProviders[providerName]; exists {
		providerMutex.RUnlock()
		return factory(config), nil
	}
	providerMutex.RUnlock()

	resilienceConfig := providers.ResolveResilienceConfig(config.Resilience)
	config.Resilience = resilienceConfig

	// If HTTPClient is not provided, inject a resilient client with retry/backoff.
	if config.HTTPClient == nil {
		config.HTTPClient = NewResilientHTTPClient(resilienceConfig)
	}

	// Check builtin providers from registry
	if factory, ok := providers.GetBuiltin(providerName); ok {
		return factory(config), nil
	}

	return nil, fmt.Errorf("unknown provider: %s", providerName)
}

// RegisterProvider registers a custom provider factory
// Returns an error if the name is empty or factory is nil
func RegisterProvider(name string, factory ProviderFactory) error {
	providerName := normalizeProviderName(name)
	if providerName == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	if factory == nil {
		return fmt.Errorf("provider factory cannot be nil")
	}

	providerMutex.Lock()
	defer providerMutex.Unlock()
	customProviders[providerName] = factory
	return nil
}

// ListRegisteredProviders returns all registered provider names
func ListRegisteredProviders() []string {
	// Get builtin providers from registry
	builtIn := providers.ListBuiltins()

	// Add custom providers
	providerMutex.RLock()
	defer providerMutex.RUnlock()
	for name := range customProviders {
		builtIn = append(builtIn, name)
	}

	return builtIn
}

// IsProviderRegistered checks if a provider is registered (built-in or custom)
func IsProviderRegistered(name string) bool {
	providerName := normalizeProviderName(name)
	if providers.IsBuiltinRegistered(providerName) {
		return true
	}

	// Check custom providers
	providerMutex.RLock()
	defer providerMutex.RUnlock()
	_, exists := customProviders[providerName]
	return exists
}
