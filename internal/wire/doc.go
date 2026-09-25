// Package wire holds what provider adapters share to speak vendor wire
// protocols: HTTP transport and Server-Sent Events, error and stop-reason
// classification, stream block indexing, ProviderOptions merging, replay
// state, and token and data URL helpers. Vendor-specific mappings live in the
// provider packages, except those shared by several, such as internal/claude.
package wire
