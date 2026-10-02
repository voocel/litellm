// Package catalog holds model facts (context window, output limit,
// reasoning support and per-token prices) loaded from LiteLLM's model list or
// set explicitly. Providers never read it; applications use it to choose
// limits and to price usage.
//
// Names are the list's keys, whose vendor prefixes follow LiteLLM's provider
// names: "xai/grok-4", not "grok/grok-4". The catalog does not translate
// names, since a model listed under several sites may be priced differently
// on each; an application keeps the catalog name of each model it prices.
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"maps"
	"math"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/voocel/litellm"
)

// DefaultURL is LiteLLM's community-maintained model list.
const DefaultURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

// Model holds what is known about a model; zero values are unknown.
type Model struct {
	// Provider is the provider the list files the model under, such as
	// "anthropic" or "vertex_ai-language-models".
	Provider string `json:"provider,omitempty"`
	// MaxInputTokens is the context window.
	MaxInputTokens  int `json:"max_input_tokens,omitempty"`
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`
	// Reasoning is nil when support is unknown; false means unsupported.
	Reasoning *bool `json:"reasoning,omitempty"`
	// Pricing is nil when the model is unpriced.
	Pricing *Pricing `json:"pricing,omitempty"`
}

// Pricing holds per-token rates; LiteLLM's list is in USD. A nil cache rate
// inherits the input rate; a non-nil zero means free cache usage.
type Pricing struct {
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

// Catalog is a concurrency-safe model table. The zero value is empty and
// ready to use.
type Catalog struct {
	mu     sync.RWMutex
	models map[string]Model
}

// Set adds or replaces the facts for the model called name.
func (c *Catalog) Set(name string, model Model) error {
	if err := model.validate(name); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.models == nil {
		c.models = make(map[string]Model)
	}
	c.models[name] = model.clone()
	return nil
}

// Get returns the facts for the exact catalog key name. A missing key returns
// a zero Model and false; provider prefixes are never added or removed.
func (c *Catalog) Get(name string) (Model, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	model, ok := c.models[name]
	return model.clone(), ok
}

// All yields the models in name order, as of the start of the iteration.
func (c *Catalog) All() iter.Seq2[string, Model] {
	return func(yield func(string, Model) bool) {
		c.mu.RLock()
		models := maps.Clone(c.models)
		c.mu.RUnlock()
		for _, name := range slices.Sorted(maps.Keys(models)) {
			if !yield(name, models[name].clone()) {
				return
			}
		}
	}
}

// LoadFromURL replaces the table with the model list at url, such as
// DefaultURL.
func (c *Catalog) LoadFromURL(ctx context.Context, url string) error {
	if strings.TrimSpace(url) == "" {
		return fmt.Errorf("catalog: url is required")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("catalog: create request: %w", err)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("catalog: fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("catalog: fetch HTTP %d", resp.StatusCode)
	}
	return c.LoadFromReader(resp.Body)
}

// LoadFromReader replaces the table with the chat and responses models of a
// model list in LiteLLM's format. A model is priced when it has both input and
// output rates. Invalid models fail the load without changing the table.
func (c *Catalog) LoadFromReader(reader io.Reader) error {
	models, err := parse(reader)
	if err != nil {
		return err
	}
	c.mu.Lock()
	c.models = models
	c.mu.Unlock()
	return nil
}

func parse(reader io.Reader) (map[string]Model, error) {
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(reader).Decode(&raw); err != nil {
		return nil, fmt.Errorf("catalog: decode model list: %w", err)
	}
	models := make(map[string]Model, len(raw))
	for name, data := range raw {
		if name == "sample_spec" { // documents the format with placeholder values
			continue
		}
		var entry struct {
			Mode            string   `json:"mode"`
			Provider        string   `json:"litellm_provider"`
			MaxInputTokens  int      `json:"max_input_tokens"`
			MaxOutputTokens int      `json:"max_output_tokens"`
			Reasoning       *bool    `json:"supports_reasoning"`
			Input           *float64 `json:"input_cost_per_token"`
			Output          *float64 `json:"output_cost_per_token"`
			CacheRead       *float64 `json:"cache_read_input_token_cost"`
			CacheWrite      *float64 `json:"cache_creation_input_token_cost"`
		}
		if err := json.Unmarshal(data, &entry); err != nil {
			return nil, fmt.Errorf("catalog: decode model %q: %w", name, err)
		}
		if entry.Mode != "chat" && entry.Mode != "responses" {
			continue
		}
		model := Model{
			Provider:        entry.Provider,
			MaxInputTokens:  entry.MaxInputTokens,
			MaxOutputTokens: entry.MaxOutputTokens,
			Reasoning:       entry.Reasoning,
		}
		if entry.Input != nil && entry.Output != nil {
			model.Pricing = &Pricing{
				InputCostPerToken:      *entry.Input,
				OutputCostPerToken:     *entry.Output,
				CacheReadCostPerToken:  entry.CacheRead,
				CacheWriteCostPerToken: entry.CacheWrite,
			}
		}
		if err := model.validate(name); err != nil {
			return nil, err
		}
		models[name] = model
	}
	return models, nil
}

// Cost prices usage. Cache reads and writes are priced at their rates and
// the rest of the input at the input rate; a cache count the vendor did not
// report is zero, its tokens priced as input.
func (p Pricing) Cost(usage litellm.Usage) (Cost, error) {
	if err := p.validate(); err != nil {
		return Cost{}, fmt.Errorf("catalog: %w", err)
	}
	cacheReadRate, cacheWriteRate := p.InputCostPerToken, p.InputCostPerToken
	if p.CacheReadCostPerToken != nil {
		cacheReadRate = *p.CacheReadCostPerToken
	}
	if p.CacheWriteCostPerToken != nil {
		cacheWriteRate = *p.CacheWriteCostPerToken
	}
	in, out, cacheRead, cacheWrite := usage.InputTokens, usage.OutputTokens, usage.CacheReadTokens, usage.CacheWriteTokens
	if in < 0 || out < 0 || cacheRead < 0 || cacheWrite < 0 || cacheRead+cacheWrite > in {
		return Cost{}, fmt.Errorf("catalog: invalid token counts: cache reads and writes must fit within input tokens")
	}
	inputCost := float64(in-cacheRead-cacheWrite) * p.InputCostPerToken
	outputCost := float64(out) * p.OutputCostPerToken
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

func (m Model) validate(name string) error {
	if strings.TrimSpace(name) == "" {
		return fmt.Errorf("catalog: model name is required")
	}
	if m.MaxInputTokens < 0 || m.MaxOutputTokens < 0 {
		return fmt.Errorf("catalog: model %q: token limits must be non-negative", name)
	}
	if m.Pricing == nil {
		return nil
	}
	if err := m.Pricing.validate(); err != nil {
		return fmt.Errorf("catalog: model %q: %w", name, err)
	}
	return nil
}

func (p Pricing) validate() error {
	for _, rate := range []struct {
		name  string
		value *float64
	}{
		{"input", &p.InputCostPerToken}, {"output", &p.OutputCostPerToken},
		{"cache read", p.CacheReadCostPerToken}, {"cache write", p.CacheWriteCostPerToken},
	} {
		if rate.value != nil && (*rate.value < 0 || math.IsNaN(*rate.value) || math.IsInf(*rate.value, 0)) {
			return fmt.Errorf("%s cost per token must be finite and non-negative", rate.name)
		}
	}
	return nil
}

// clone keeps optional facts and rates owned by the catalog or its caller.
func (m Model) clone() Model {
	if m.Reasoning != nil {
		m.Reasoning = new(*m.Reasoning)
	}
	if m.Pricing == nil {
		return m
	}
	pricing := *m.Pricing
	pricing.CacheReadCostPerToken = copyRate(pricing.CacheReadCostPerToken)
	pricing.CacheWriteCostPerToken = copyRate(pricing.CacheWriteCostPerToken)
	m.Pricing = &pricing
	return m
}

func copyRate(rate *float64) *float64 {
	if rate == nil {
		return nil
	}
	return new(*rate)
}
