package providers

// BuiltinFactory is a function that creates a Provider instance
type BuiltinFactory func(ProviderConfig) Provider

// builtinRegistry stores all registered builtin providers
var builtinRegistry = make(map[string]BuiltinFactory)

// defaultURLs stores default base URLs for each provider
var defaultURLs = make(map[string]string)

// RegisterBuiltin registers a builtin provider with its factory and default URL.
// This should be called in each provider's init() function.
func RegisterBuiltin(name string, factory BuiltinFactory, defaultURL string) {
	builtinRegistry[name] = factory
	defaultURLs[name] = defaultURL
}

// GetBuiltin returns the factory function for a builtin provider.
// Returns nil and false if the provider is not registered.
func GetBuiltin(name string) (BuiltinFactory, bool) {
	factory, ok := builtinRegistry[name]
	return factory, ok
}

// ListBuiltins returns a list of all registered builtin provider names.
func ListBuiltins() []string {
	names := make([]string, 0, len(builtinRegistry))
	for name := range builtinRegistry {
		names = append(names, name)
	}
	return names
}

// GetDefaultURL returns the default base URL for a provider.
// Returns empty string if the provider is not registered.
func GetDefaultURL(name string) string {
	return defaultURLs[name]
}

// IsBuiltinRegistered checks if a provider is registered as a builtin.
func IsBuiltinRegistered(name string) bool {
	_, ok := builtinRegistry[name]
	return ok
}
