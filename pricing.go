package litellm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const PricingURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

// ModelPricing contains pricing information for a model.
type ModelPricing struct {
	InputCostPerToken      float64 `json:"input_cost_per_token"`
	OutputCostPerToken     float64 `json:"output_cost_per_token"`
	CacheReadCostPerToken  float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheWriteCostPerToken float64 `json:"cache_creation_input_token_cost,omitempty"`
}

// ModelCapabilities contains capability and limit metadata for a model.
type ModelCapabilities struct {
	Provider          string `json:"litellm_provider"`
	MaxInputTokens    int    `json:"max_input_tokens"`
	MaxOutputTokens   int    `json:"max_output_tokens"`
	SupportsTools     bool   `json:"supports_function_calling"`
	SupportsVision    bool   `json:"supports_vision"`
	SupportsReasoning bool   `json:"supports_reasoning"`
}

// CostResult contains the calculated cost for a request.
type CostResult struct {
	InputCost     float64 `json:"input_cost"`
	OutputCost    float64 `json:"output_cost"`
	CacheReadCost float64 `json:"cache_read_cost,omitempty"`
	CacheWriteCost float64 `json:"cache_write_cost,omitempty"`
	TotalCost     float64 `json:"total_cost"`
	Currency      string  `json:"currency"`
}

// modelEntry is the full internal record parsed from the registry JSON.
type modelEntry struct {
	inputCostPerToken      float64
	outputCostPerToken     float64
	cacheReadCostPerToken  float64
	cacheWriteCostPerToken float64
	hasInputPricing        bool
	hasOutputPricing       bool
	provider               string
	maxInputTokens         int
	maxOutputTokens        int
	supportsTools          bool
	supportsVision         bool
	supportsReasoning      bool
}

var (
	registryData   = make(map[string]modelEntry)
	registryMu     sync.RWMutex
	registryLoaded bool
)

// LoadPricing fetches model registry data from BerriAI/litellm GitHub.
func LoadPricing(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", PricingURL, nil)
	if err != nil {
		return fmt.Errorf("create pricing request: %w", err)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("fetch pricing: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch pricing: HTTP %d", resp.StatusCode)
	}

	return LoadPricingFromReader(resp.Body)
}

// LoadPricingFromReader loads model registry data from any io.Reader.
func LoadPricingFromReader(r io.Reader) error {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return fmt.Errorf("decode pricing: %w", err)
	}

	data := make(map[string]modelEntry, len(raw))
	for model, rawData := range raw {
		if model == "sample_spec" {
			continue
		}

		var parsed struct {
			InputCostPerToken      *float64 `json:"input_cost_per_token"`
			OutputCostPerToken     *float64 `json:"output_cost_per_token"`
			CacheReadCostPerToken  *float64 `json:"cache_read_input_token_cost"`
			CacheWriteCostPerToken *float64 `json:"cache_creation_input_token_cost"`
			Provider               string   `json:"litellm_provider"`
			MaxInputTokens         int      `json:"max_input_tokens"`
			MaxOutputTokens        int      `json:"max_output_tokens"`
			SupportsTools          bool     `json:"supports_function_calling"`
			SupportsVision         bool     `json:"supports_vision"`
			SupportsReasoning      bool     `json:"supports_reasoning"`
		}
		if err := json.Unmarshal(rawData, &parsed); err != nil {
			continue
		}

		entry := modelEntry{
			provider:          parsed.Provider,
			maxInputTokens:    parsed.MaxInputTokens,
			maxOutputTokens:   parsed.MaxOutputTokens,
			supportsTools:     parsed.SupportsTools,
			supportsVision:    parsed.SupportsVision,
			supportsReasoning: parsed.SupportsReasoning,
		}
		if parsed.InputCostPerToken != nil {
			entry.inputCostPerToken = *parsed.InputCostPerToken
			entry.hasInputPricing = true
		}
		if parsed.OutputCostPerToken != nil {
			entry.outputCostPerToken = *parsed.OutputCostPerToken
			entry.hasOutputPricing = true
		}
		if parsed.CacheReadCostPerToken != nil {
			entry.cacheReadCostPerToken = *parsed.CacheReadCostPerToken
		}
		if parsed.CacheWriteCostPerToken != nil {
			entry.cacheWriteCostPerToken = *parsed.CacheWriteCostPerToken
		}
		data[model] = entry
	}

	registryMu.Lock()
	registryData = data
	registryLoaded = true
	registryMu.Unlock()

	return nil
}

// GetModelPricing returns the pricing for a model.
func GetModelPricing(model string) (*ModelPricing, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()

	if !registryLoaded {
		return nil, false
	}
	entry, ok := registryData[model]
	if !ok {
		return nil, false
	}
	if !entry.hasInputPricing || !entry.hasOutputPricing {
		return nil, false
	}
	return &ModelPricing{
		InputCostPerToken:      entry.inputCostPerToken,
		OutputCostPerToken:     entry.outputCostPerToken,
		CacheReadCostPerToken:  entry.cacheReadCostPerToken,
		CacheWriteCostPerToken: entry.cacheWriteCostPerToken,
	}, true
}

// GetModelCapabilities returns capability metadata for a model.
func GetModelCapabilities(model string) (*ModelCapabilities, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()

	if !registryLoaded {
		return nil, false
	}
	entry, ok := registryData[model]
	if !ok {
		return nil, false
	}
	return &ModelCapabilities{
		Provider:          entry.provider,
		MaxInputTokens:    entry.maxInputTokens,
		MaxOutputTokens:   entry.maxOutputTokens,
		SupportsTools:     entry.supportsTools,
		SupportsVision:    entry.supportsVision,
		SupportsReasoning: entry.supportsReasoning,
	}, true
}

// SetModelPricing sets custom pricing for a model.
func SetModelPricing(model string, pricing ModelPricing) {
	registryMu.Lock()
	defer registryMu.Unlock()
	entry := registryData[model]
	entry.inputCostPerToken = pricing.InputCostPerToken
	entry.outputCostPerToken = pricing.OutputCostPerToken
	entry.cacheReadCostPerToken = pricing.CacheReadCostPerToken
	entry.cacheWriteCostPerToken = pricing.CacheWriteCostPerToken
	entry.hasInputPricing = true
	entry.hasOutputPricing = true
	registryData[model] = entry
	registryLoaded = true
}

// CalculateCost calculates the cost based on token usage.
// Registry data is loaded automatically on first call.
func CalculateCost(model string, usage Usage) (*CostResult, error) {
	if !IsPricingLoaded() {
		if err := LoadPricing(context.Background()); err != nil {
			return nil, err
		}
	}

	pricing, ok := GetModelPricing(model)
	if !ok {
		return nil, fmt.Errorf("pricing not found for model: %s", model)
	}

	inputCost := float64(usage.PromptTokens) * pricing.InputCostPerToken
	outputCost := float64(usage.CompletionTokens) * pricing.OutputCostPerToken

	// Cache costs: use dedicated rates when available, fall back to input rate
	cacheReadRate := pricing.CacheReadCostPerToken
	if cacheReadRate == 0 {
		cacheReadRate = pricing.InputCostPerToken
	}
	cacheWriteRate := pricing.CacheWriteCostPerToken
	if cacheWriteRate == 0 {
		cacheWriteRate = pricing.InputCostPerToken
	}
	cacheReadCost := float64(usage.CacheReadInputTokens) * cacheReadRate
	cacheWriteCost := float64(usage.CacheCreationInputTokens) * cacheWriteRate

	return &CostResult{
		InputCost:      inputCost,
		OutputCost:     outputCost,
		CacheReadCost:  cacheReadCost,
		CacheWriteCost: cacheWriteCost,
		TotalCost:      inputCost + outputCost + cacheReadCost + cacheWriteCost,
		Currency:       "USD",
	}, nil
}

// CalculateCostForResponse calculates the cost for a response.
func CalculateCostForResponse(resp *Response) (*CostResult, error) {
	if resp == nil {
		return nil, fmt.Errorf("response is nil")
	}
	return CalculateCost(resp.Model, resp.Usage)
}

// IsPricingLoaded returns whether registry data has been loaded.
func IsPricingLoaded() bool {
	registryMu.RLock()
	defer registryMu.RUnlock()
	return registryLoaded
}
