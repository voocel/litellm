package litellm

import (
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/voocel/litellm/providers"
)

// Global registry for custom providers
var (
	customProviders = make(map[string]ProviderDescriptor)
	providerMutex   sync.RWMutex
)

func normalizeProviderName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

// createProvider creates a provider instance by name
func createProvider(name string, config ProviderConfig) (Provider, error) {
	providerName := normalizeProviderName(name)
	config = normalizeProviderConfig(config)

	// Check custom providers first
	providerMutex.RLock()
	if descriptor, exists := customProviders[providerName]; exists {
		providerMutex.RUnlock()
		if config.BaseURL == "" && descriptor.DefaultURL != "" {
			config.BaseURL = descriptor.DefaultURL
		}
		return descriptor.Factory(config), nil
	}
	providerMutex.RUnlock()

	// Check builtin providers from registry
	if factory, ok := providers.GetBuiltin(providerName); ok {
		return factory(config), nil
	}

	return nil, fmt.Errorf("unknown provider: %s", providerName)
}

func normalizeProviderConfig(config ProviderConfig) ProviderConfig {
	resilienceConfig := providers.ResolveResilienceConfig(config.Resilience)
	if config.Timeout > 0 {
		resilienceConfig.RequestTimeout = config.Timeout
	}
	config.Resilience = resilienceConfig

	if config.HTTPClient == nil {
		config.HTTPClient = NewResilientHTTPClient(resilienceConfig)
	}

	return config
}

// RegisterProvider registers a custom provider factory
// Returns an error if the name is empty or factory is nil
func RegisterProvider(name string, factory ProviderFactory) error {
	return RegisterProviderWithDescriptor(ProviderDescriptor{
		Name:    name,
		Factory: factory,
	})
}

// RegisterProviderWithDescriptor registers a custom provider together with
// lightweight static metadata such as its default BaseURL.
func RegisterProviderWithDescriptor(descriptor ProviderDescriptor) error {
	providerName := normalizeProviderName(descriptor.Name)
	if providerName == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	if descriptor.Factory == nil {
		return fmt.Errorf("provider factory cannot be nil")
	}

	providerMutex.Lock()
	defer providerMutex.Unlock()
	if providers.IsBuiltinRegistered(providerName) {
		return fmt.Errorf("provider %q conflicts with builtin provider", providerName)
	}
	if _, exists := customProviders[providerName]; exists {
		return fmt.Errorf("provider %q is already registered", providerName)
	}

	descriptor.Name = providerName
	descriptor.DefaultURL = strings.TrimSpace(descriptor.DefaultURL)
	customProviders[providerName] = descriptor
	return nil
}

// ListRegisteredProviders returns all registered provider names
func ListRegisteredProviders() []string {
	names := make([]string, 0, len(providers.ListBuiltins())+len(customProviders))

	for _, name := range providers.ListBuiltins() {
		names = append(names, name)
	}

	providerMutex.RLock()
	defer providerMutex.RUnlock()
	for name := range customProviders {
		names = append(names, name)
	}

	slices.Sort(names)
	return names
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
