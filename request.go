package litellm

import "github.com/voocel/litellm/providers"

// Core types are sourced from providers; litellm re-exports them.
type (
	Message         = providers.Message
	MessageContent  = providers.MessageContent
	MessageImageURL = providers.MessageImageURL
	CacheControl    = providers.CacheControl

	Tool          = providers.Tool
	FunctionDef   = providers.FunctionDef
	ToolCall      = providers.ToolCall
	FunctionCall  = providers.FunctionCall
	ToolCallDelta = providers.ToolCallDelta

	ResponseFormat         = providers.ResponseFormat
	JSONSchema             = providers.JSONSchema
	OpenAIResponsesRequest = providers.OpenAIResponsesRequest

	Request       = providers.Request
	Response      = providers.Response
	Usage         = providers.Usage
	ReasoningData = providers.ReasoningData

	StreamChunk    = providers.StreamChunk
	ReasoningChunk = providers.ReasoningChunk
	StreamReader   = providers.StreamReader
)

// CacheControl type constants.
const (
	CacheTypeEphemeral  = "ephemeral"
	CacheTypePersistent = "persistent"
)

// Stream chunk type constants.
const (
	ChunkTypeContent       = "content"
	ChunkTypeToolCallDelta = "tool_call_delta"
	ChunkTypeReasoning     = "reasoning"
)

// ResponseFormat type constants.
const (
	ResponseFormatText       = "text"
	ResponseFormatJSONObject = "json_object"
	ResponseFormatJSONSchema = "json_schema"
)

// NewEphemeralCache creates an ephemeral cache control (TTL is provider-defined, typically ~5 minutes).
func NewEphemeralCache() *CacheControl {
	return &CacheControl{Type: CacheTypeEphemeral}
}

// NewPersistentCache creates a persistent cache control with a custom TTL (seconds).
func NewPersistentCache(ttlSeconds int) *CacheControl {
	return &CacheControl{
		Type: CacheTypePersistent,
		TTL:  &ttlSeconds,
	}
}

// NewCacheControl creates a cache control with optional TTL.
func NewCacheControl(cacheType string, ttlSeconds ...int) *CacheControl {
	cache := &CacheControl{Type: cacheType}
	if len(ttlSeconds) > 0 {
		cache.TTL = &ttlSeconds[0]
	}
	return cache
}
