// Package pricing computes request cost from Usage and per-token rates, set
// explicitly or loaded into a Registry from LiteLLM's price list.
package pricing

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/voocel/litellm"
)

// DefaultURL is LiteLLM's community-maintained model price list.
const DefaultURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

// ModelPricing holds per-token rates; LiteLLM's list is in USD. A nil cache
// rate inherits the input rate; a non-nil zero means free cache usage.
type ModelPricing struct {
	InputCostPerToken      float64  `json:"input_cost_per_token"`
	OutputCostPerToken     float64  `json:"output_cost_per_token"`
	CacheReadCostPerToken  *float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheWriteCostPerToken *float64 `json:"cache_creation_input_token_cost,omitempty"`
}

// Cost is a cost breakdown in the currency of the rates.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read,omitempty"`
	CacheWrite float64 `json:"cache_write,omitempty"`
	Total      float64 `json:"total"`
}

// Registry is a concurrency-safe price table. The zero value is ready to use.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]ModelPricing
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{entries: make(map[string]ModelPricing)}
}

// Set adds or replaces the rates for model.
func (r *Registry) Set(model string, price ModelPricing) error {
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("pricing: model is required")
	}
	if err := validatePricing(price); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.entries == nil {
		r.entries = make(map[string]ModelPricing)
	}
	r.entries[model] = clonePricing(price)
	return nil
}

// Get returns the rates for model. A "vendor/model" name falls back to "model"
// when it has no entry of its own.
func (r *Registry) Get(model string) (ModelPricing, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	price, ok := r.lookup(model)
	return clonePricing(price), ok
}

// Cost prices usage with the rates for model.
func (r *Registry) Cost(model string, usage litellm.Usage) (Cost, error) {
	price, ok := r.Get(model)
	if !ok {
		return Cost{}, fmt.Errorf("pricing: model %q has no pricing", model)
	}
	return price.Cost(usage)
}

// LoadFromURL replaces the table with the JSON price list at url, such as
// DefaultURL.
func (r *Registry) LoadFromURL(ctx context.Context, url string) error {
	if strings.TrimSpace(url) == "" {
		return fmt.Errorf("pricing: url is required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("pricing: create request: %w", err)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("pricing: fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("pricing: fetch HTTP %d", resp.StatusCode)
	}
	return r.LoadFromReader(resp.Body)
}

// LoadFromReader replaces the table with a JSON price list in LiteLLM's
// format. Models without both input and output rates are skipped.
func (r *Registry) LoadFromReader(reader io.Reader) error {
	entries, err := parseRegistry(reader)
	if err != nil {
		return err
	}
	r.mu.Lock()
	r.entries = entries
	r.mu.Unlock()
	return nil
}

// Cost prices usage. Input and output counts must be known; cache counts may
// be unknown only when their rate equals the input rate.
func (price ModelPricing) Cost(usage litellm.Usage) (Cost, error) {
	if err := validatePricing(price); err != nil {
		return Cost{}, err
	}
	if usage.InputTokens == nil || usage.OutputTokens == nil {
		return Cost{}, fmt.Errorf("pricing: input and output token counts must be known")
	}
	cacheReadRate, cacheWriteRate := price.InputCostPerToken, price.InputCostPerToken
	if price.CacheReadCostPerToken != nil {
		cacheReadRate = *price.CacheReadCostPerToken
	}
	if price.CacheWriteCostPerToken != nil {
		cacheWriteRate = *price.CacheWriteCostPerToken
	}
	cacheRead, cacheWrite := 0, 0
	if usage.CacheReadTokens != nil {
		cacheRead = *usage.CacheReadTokens
	} else if cacheReadRate != price.InputCostPerToken {
		return Cost{}, fmt.Errorf("pricing: cache read token count is unknown for a distinct cache rate")
	}
	if usage.CacheWriteTokens != nil {
		cacheWrite = *usage.CacheWriteTokens
	} else if cacheWriteRate != price.InputCostPerToken {
		return Cost{}, fmt.Errorf("pricing: cache write token count is unknown for a distinct cache rate")
	}
	if *usage.InputTokens < 0 || *usage.OutputTokens < 0 || cacheRead < 0 || cacheWrite < 0 ||
		cacheRead > *usage.InputTokens || cacheWrite > *usage.InputTokens-cacheRead {
		return Cost{}, fmt.Errorf("pricing: invalid token counts: cache reads and writes must fit within input tokens")
	}
	// Unknown cache details stay in Input when all applicable rates are equal.
	nonCachedInput := *usage.InputTokens - cacheRead - cacheWrite
	inputCost := float64(nonCachedInput) * price.InputCostPerToken
	outputCost := float64(*usage.OutputTokens) * price.OutputCostPerToken
	cacheReadCost := float64(cacheRead) * cacheReadRate
	cacheWriteCost := float64(cacheWrite) * cacheWriteRate

	return Cost{
		Input:      inputCost,
		Output:     outputCost,
		CacheRead:  cacheReadCost,
		CacheWrite: cacheWriteCost,
		Total:      inputCost + outputCost + cacheReadCost + cacheWriteCost,
	}, nil
}

func parseRegistry(reader io.Reader) (map[string]ModelPricing, error) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(reader).Decode(&raw); err != nil {
		return nil, fmt.Errorf("pricing: decode registry: %w", err)
	}
	entries := make(map[string]ModelPricing, len(raw))
	for model, rawData := range raw {
		if model == "sample_spec" {
			continue
		}
		var parsed struct {
			InputCostPerToken      *float64 `json:"input_cost_per_token"`
			OutputCostPerToken     *float64 `json:"output_cost_per_token"`
			CacheReadCostPerToken  *float64 `json:"cache_read_input_token_cost"`
			CacheWriteCostPerToken *float64 `json:"cache_creation_input_token_cost"`
		}
		if err := json.Unmarshal(rawData, &parsed); err != nil {
			return nil, fmt.Errorf("pricing: decode model %q: %w", model, err)
		}
		if parsed.InputCostPerToken == nil || parsed.OutputCostPerToken == nil {
			continue
		}
		entries[model] = ModelPricing{
			InputCostPerToken:      *parsed.InputCostPerToken,
			OutputCostPerToken:     *parsed.OutputCostPerToken,
			CacheReadCostPerToken:  parsed.CacheReadCostPerToken,
			CacheWriteCostPerToken: parsed.CacheWriteCostPerToken,
		}
	}
	return entries, nil
}

func (r *Registry) lookup(model string) (ModelPricing, bool) {
	if r == nil {
		return ModelPricing{}, false
	}
	if price, ok := r.entries[model]; ok {
		return price, true
	}
	if _, after, ok := strings.Cut(model, "/"); ok {
		price, found := r.entries[after]
		return price, found
	}
	return ModelPricing{}, false
}

// clonePricing copies the optional cache rates, which distinguish an inherited
// rate (nil) from free usage (zero).
func clonePricing(price ModelPricing) ModelPricing {
	price.CacheReadCostPerToken = copyRate(price.CacheReadCostPerToken)
	price.CacheWriteCostPerToken = copyRate(price.CacheWriteCostPerToken)
	return price
}

func copyRate(rate *float64) *float64 {
	if rate == nil {
		return nil
	}
	value := *rate
	return &value
}

func validatePricing(price ModelPricing) error {
	for _, rate := range []struct {
		name  string
		value *float64
	}{
		{"input", &price.InputCostPerToken}, {"output", &price.OutputCostPerToken},
		{"cache read", price.CacheReadCostPerToken}, {"cache write", price.CacheWriteCostPerToken},
	} {
		if rate.value != nil && (*rate.value < 0 || math.IsNaN(*rate.value) || math.IsInf(*rate.value, 0)) {
			return fmt.Errorf("pricing: %s cost per token must be finite and non-negative", rate.name)
		}
	}
	return nil
}
