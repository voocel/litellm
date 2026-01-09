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
	InputCostPerToken  float64 `json:"input_cost_per_token"`
	OutputCostPerToken float64 `json:"output_cost_per_token"`
}

// CostResult contains the calculated cost for a request.
type CostResult struct {
	InputCost  float64 `json:"input_cost"`
	OutputCost float64 `json:"output_cost"`
	TotalCost  float64 `json:"total_cost"`
	Currency   string  `json:"currency"`
}

var (
	pricingData   = make(map[string]ModelPricing)
	pricingMu     sync.RWMutex
	pricingLoaded bool
)

// LoadPricing fetches pricing data from BerriAI/litellm GitHub.
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

// LoadPricingFromReader loads pricing data from any io.Reader.
func LoadPricingFromReader(r io.Reader) error {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(r).Decode(&raw); err != nil {
		return fmt.Errorf("decode pricing: %w", err)
	}

	data := make(map[string]ModelPricing)
	for model, rawData := range raw {
		if model == "sample_spec" {
			continue
		}

		var pricing struct {
			InputCostPerToken  *float64 `json:"input_cost_per_token"`
			OutputCostPerToken *float64 `json:"output_cost_per_token"`
		}
		if err := json.Unmarshal(rawData, &pricing); err != nil {
			continue
		}

		if pricing.InputCostPerToken != nil && pricing.OutputCostPerToken != nil {
			data[model] = ModelPricing{
				InputCostPerToken:  *pricing.InputCostPerToken,
				OutputCostPerToken: *pricing.OutputCostPerToken,
			}
		}
	}

	pricingMu.Lock()
	pricingData = data
	pricingLoaded = true
	pricingMu.Unlock()

	return nil
}

// GetModelPricing returns the pricing for a model.
func GetModelPricing(model string) (*ModelPricing, bool) {
	pricingMu.RLock()
	defer pricingMu.RUnlock()

	if !pricingLoaded {
		return nil, false
	}
	pricing, ok := pricingData[model]
	if !ok {
		return nil, false
	}
	return &pricing, true
}

// SetModelPricing sets custom pricing for a model.
func SetModelPricing(model string, pricing ModelPricing) {
	pricingMu.Lock()
	defer pricingMu.Unlock()
	pricingData[model] = pricing
	pricingLoaded = true
}

// CalculateCost calculates the cost based on token usage.
// Pricing data is loaded automatically on first call.
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

	return &CostResult{
		InputCost:  inputCost,
		OutputCost: outputCost,
		TotalCost:  inputCost + outputCost,
		Currency:   "USD",
	}, nil
}

// CalculateCostForResponse calculates the cost for a response.
func CalculateCostForResponse(resp *Response) (*CostResult, error) {
	if resp == nil {
		return nil, fmt.Errorf("response is nil")
	}
	return CalculateCost(resp.Model, resp.Usage)
}

// IsPricingLoaded returns whether pricing data has been loaded.
func IsPricingLoaded() bool {
	pricingMu.RLock()
	defer pricingMu.RUnlock()
	return pricingLoaded
}
