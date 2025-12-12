package litellm

import "github.com/voocel/litellm/providers"

// Provider and ProviderConfig are sourced from providers; re-exported here.
type Provider = providers.Provider
type ProviderConfig = providers.ProviderConfig

// ProviderFactory is used to register custom providers.
type ProviderFactory func(config ProviderConfig) Provider
