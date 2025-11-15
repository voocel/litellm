package litellm

import (
	"context"
	"fmt"
	"slices"
	"sync"

	"github.com/voocel/litellm/providers"
)

// Global registry for custom providers
var (
	customProviders = make(map[string]ProviderFactory)
	providerMutex   sync.RWMutex
)

// createProvider creates a provider instance by name
func createProvider(name string, config ProviderConfig) (Provider, error) {
	// Check custom providers first
	providerMutex.RLock()
	if factory, exists := customProviders[name]; exists {
		providerMutex.RUnlock()
		return factory(config), nil
	}
	providerMutex.RUnlock()

	resilienceConfig := config.Resilience
	if resilienceConfig == (ResilienceConfig{}) {
		resilienceConfig = DefaultResilienceConfig()
	}

	resilientClient := NewResilientHTTPClient(resilienceConfig)

	// Fall back to built-in providers
	providerConfig := providers.ProviderConfig{
		APIKey:  config.APIKey,
		BaseURL: config.BaseURL,
		Extra:   config.Extra,
		Resilience: providers.ResilienceConfig{
			MaxRetries:     resilienceConfig.MaxRetries,
			InitialDelay:   resilienceConfig.InitialDelay,
			MaxDelay:       resilienceConfig.MaxDelay,
			Multiplier:     resilienceConfig.Multiplier,
			Jitter:         resilienceConfig.Jitter,
			RequestTimeout: resilienceConfig.RequestTimeout,
			ConnectTimeout: resilienceConfig.ConnectTimeout,
		},
		HTTPClient: resilientClient,
	}

	// Create provider directly based on name
	switch name {
	case "openai":
		p := providers.NewOpenAI(providerConfig)
		return &providerAdapter{p}, nil
	case "anthropic":
		p := providers.NewAnthropic(providerConfig)
		return &providerAdapter{p}, nil
	case "gemini":
		p := providers.NewGemini(providerConfig)
		return &providerAdapter{p}, nil
	case "deepseek":
		p := providers.NewDeepSeek(providerConfig)
		return &providerAdapter{p}, nil
	case "glm":
		p := providers.NewGLM(providerConfig)
		return &providerAdapter{p}, nil
	case "openrouter":
		p := providers.NewOpenRouter(providerConfig)
		return &providerAdapter{p}, nil
	case "qwen":
		p := providers.NewQwen(providerConfig)
		return &providerAdapter{p}, nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", name)
	}
}

// providerAdapter adapts providers.Provider to main package Provider interface
type providerAdapter struct {
	provider providers.Provider
}

func (a *providerAdapter) Name() string {
	return a.provider.Name()
}

func (a *providerAdapter) Validate() error {
	return a.provider.Validate()
}

func (a *providerAdapter) SupportsModel(model string) bool {
	return a.provider.SupportsModel(model)
}

func (a *providerAdapter) Models() []ModelInfo {
	providerModels := a.provider.Models()
	models := make([]ModelInfo, len(providerModels))
	for i, m := range providerModels {
		capabilities := make([]ModelCapability, len(m.Capabilities))
		for j, cap := range m.Capabilities {
			capabilities[j] = ModelCapability(cap)
		}
		models[i] = ModelInfo{
			ID:           m.ID,
			Provider:     m.Provider,
			Name:         m.Name,
			MaxTokens:    m.MaxTokens,
			Capabilities: capabilities,
		}
	}
	return models
}

func (a *providerAdapter) Chat(ctx context.Context, req *Request) (*Response, error) {
	providerReq := &providers.Request{
		Model:            req.Model,
		Messages:         convertMessages(req.Messages),
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		Stream:           req.Stream,
		Tools:            convertTools(req.Tools),
		ToolChoice:       req.ToolChoice,
		ResponseFormat:   convertResponseFormat(req.ResponseFormat),
		Stop:             req.Stop,
		ReasoningEffort:  req.ReasoningEffort,
		ReasoningSummary: req.ReasoningSummary,
		UseResponsesAPI:  req.UseResponsesAPI,
		Extra:            req.Extra,
	}

	resp, err := a.provider.Chat(ctx, providerReq)
	if err != nil {
		return nil, err
	}

	return &Response{
		Content:      resp.Content,
		ToolCalls:    convertProviderToolCalls(resp.ToolCalls),
		Usage:        convertUsage(resp.Usage),
		Model:        resp.Model,
		Provider:     resp.Provider,
		FinishReason: resp.FinishReason,
		Reasoning:    convertReasoningData(resp.Reasoning),
	}, nil
}

func (a *providerAdapter) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	providerReq := &providers.Request{
		Model:            req.Model,
		Messages:         convertMessages(req.Messages),
		MaxTokens:        req.MaxTokens,
		Temperature:      req.Temperature,
		Stream:           true,
		Tools:            convertTools(req.Tools),
		ToolChoice:       req.ToolChoice,
		ResponseFormat:   convertResponseFormat(req.ResponseFormat),
		Stop:             req.Stop,
		ReasoningEffort:  req.ReasoningEffort,
		ReasoningSummary: req.ReasoningSummary,
		UseResponsesAPI:  req.UseResponsesAPI,
		Extra:            req.Extra,
	}

	stream, err := a.provider.Stream(ctx, providerReq)
	if err != nil {
		return nil, err
	}

	return &streamAdapter{stream}, nil
}

// streamAdapter adapts providers.StreamReader to main package StreamReader
type streamAdapter struct {
	stream providers.StreamReader
}

func (s *streamAdapter) Next() (*StreamChunk, error) {
	chunk, err := s.stream.Next()
	if err != nil {
		return nil, err
	}

	streamChunk := &StreamChunk{
		Type:         chunk.Type,
		Content:      chunk.Content,
		FinishReason: chunk.FinishReason,
		Model:        chunk.Model,
		Provider:     chunk.Provider,
		Done:         chunk.Done,
	}

	// Convert tool call delta
	if chunk.ToolCallDelta != nil {
		streamChunk.ToolCallDelta = &ToolCallDelta{
			Index:          chunk.ToolCallDelta.Index,
			ID:             chunk.ToolCallDelta.ID,
			Type:           chunk.ToolCallDelta.Type,
			FunctionName:   chunk.ToolCallDelta.FunctionName,
			ArgumentsDelta: chunk.ToolCallDelta.ArgumentsDelta,
		}
	}

	if chunk.Reasoning != nil {
		streamChunk.Reasoning = &ReasoningChunk{
			Summary: chunk.Reasoning.Summary,
			Content: chunk.Reasoning.Content,
		}
	}

	// Convert usage information
	if chunk.Usage != nil {
		streamChunk.Usage = &Usage{
			PromptTokens:     chunk.Usage.PromptTokens,
			CompletionTokens: chunk.Usage.CompletionTokens,
			TotalTokens:      chunk.Usage.TotalTokens,
			ReasoningTokens:  chunk.Usage.ReasoningTokens,
		}
	}

	return streamChunk, nil
}

func (s *streamAdapter) Close() error {
	return s.stream.Close()
}

func convertMessages(messages []Message) []providers.Message {
	providerMessages := make([]providers.Message, len(messages))
	for i, msg := range messages {
		providerMessages[i] = providers.Message{
			Role:         msg.Role,
			Content:      msg.Content,
			ToolCalls:    convertMessageToolCalls(msg.ToolCalls),
			ToolCallID:   msg.ToolCallID,
			CacheControl: convertCacheControl(msg.CacheControl),
		}
	}
	return providerMessages
}

func convertCacheControl(cache *CacheControl) *providers.CacheControl {
	if cache == nil {
		return nil
	}
	converted := &providers.CacheControl{Type: cache.Type}
	if cache.TTL != nil {
		ttl := *cache.TTL
		converted.TTL = &ttl
	}

	return converted
}

func convertMessageToolCalls(toolCalls []ToolCall) []providers.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}
	providerToolCalls := make([]providers.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		providerToolCalls[i] = providers.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: providers.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return providerToolCalls
}

func convertProviderToolCalls(toolCalls []providers.ToolCall) []ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}
	mainToolCalls := make([]ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		mainToolCalls[i] = ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return mainToolCalls
}

func convertTools(tools []Tool) []providers.Tool {
	if len(tools) == 0 {
		return nil
	}
	providerTools := make([]providers.Tool, len(tools))
	for i, tool := range tools {
		providerTools[i] = providers.Tool{
			Type: tool.Type,
			Function: providers.FunctionDef{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}
	return providerTools
}

func convertResponseFormat(rf *ResponseFormat) *providers.ResponseFormat {
	if rf == nil {
		return nil
	}
	providerRF := &providers.ResponseFormat{
		Type: rf.Type,
	}
	if rf.JSONSchema != nil {
		providerRF.JSONSchema = &providers.JSONSchema{
			Name:        rf.JSONSchema.Name,
			Description: rf.JSONSchema.Description,
			Schema:      rf.JSONSchema.Schema,
			Strict:      rf.JSONSchema.Strict,
		}
	}
	return providerRF
}

func convertUsage(usage providers.Usage) Usage {
	return Usage{
		PromptTokens:             usage.PromptTokens,
		CompletionTokens:         usage.CompletionTokens,
		TotalTokens:              usage.TotalTokens,
		ReasoningTokens:          usage.ReasoningTokens,
		CacheCreationInputTokens: usage.CacheCreationInputTokens,
		CacheReadInputTokens:     usage.CacheReadInputTokens,
	}
}

func convertReasoningData(reasoning *providers.ReasoningData) *ReasoningData {
	if reasoning == nil {
		return nil
	}
	return &ReasoningData{
		Summary:    reasoning.Summary,
		Content:    reasoning.Content,
		TokensUsed: reasoning.TokensUsed,
	}
}

// RegisterProvider registers a custom provider factory
// Returns an error if the name is empty or factory is nil
func RegisterProvider(name string, factory ProviderFactory) error {
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	if factory == nil {
		return fmt.Errorf("provider factory cannot be nil")
	}

	providerMutex.Lock()
	defer providerMutex.Unlock()
	customProviders[name] = factory
	return nil
}

// ListRegisteredProviders returns all registered provider names
func ListRegisteredProviders() []string {
	providerMutex.RLock()
	defer providerMutex.RUnlock()

	// Built-in providers
	builtIn := []string{"openai", "anthropic", "gemini", "deepseek", "glm", "openrouter", "qwen"}

	// Add custom providers
	for name := range customProviders {
		builtIn = append(builtIn, name)
	}

	return builtIn
}

// IsProviderRegistered checks if a provider is registered (built-in or custom)
func IsProviderRegistered(name string) bool {
	// Check built-in providers
	builtIn := []string{"openai", "anthropic", "gemini", "deepseek", "glm", "openrouter", "qwen"}
	if slices.Contains(builtIn, name) {
		return true
	}

	// Check custom providers
	providerMutex.RLock()
	defer providerMutex.RUnlock()
	_, exists := customProviders[name]
	return exists
}

// HasChatCapability checks if provider supports chat
func HasChatCapability(p any) bool {
	_, ok := p.(ChatProvider)
	return ok
}

// HasStreamCapability checks if provider supports streaming
func HasStreamCapability(p any) bool {
	_, ok := p.(StreamProvider)
	return ok
}

// HasModelCapability checks if provider supports model information
func HasModelCapability(p any) bool {
	_, ok := p.(ModelProvider)
	return ok
}

// SupportsCapability checks if provider supports a specific model capability
func SupportsCapability(p Provider, model string, capability ModelCapability) bool {
	if !HasModelCapability(p) {
		return false
	}

	models := p.Models()
	for _, m := range models {
		if m.ID == model {
			return slices.Contains(m.Capabilities, capability)
		}
	}
	return false
}
