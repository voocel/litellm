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
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"maps"
	"math"
	"net/http"
	"regexp"
	"slices"
	"strconv"
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

// Pricing holds a model's rates; LiteLLM's list is in USD. Vendors that
// charge more for long inputs price every token of such a call at a tier's
// rates.
type Pricing struct {
	Rates
	// Tiers, in increasing order of AboveInputTokens, replace Rates for a
	// call whose input tokens exceed AboveInputTokens: the last such tier.
	Tiers []Tier `json:"tiers,omitempty"`
}

// Rates are per-token rates. A nil cache rate inherits the input rate; a
// non-nil zero means free cache usage. Writes cached for an hour are priced
// at CacheWrite1hCostPerToken alone, since vendors charge more for them.
type Rates struct {
	InputCostPerToken        float64  `json:"input_cost_per_token"`
	OutputCostPerToken       float64  `json:"output_cost_per_token"`
	CacheReadCostPerToken    *float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheWriteCostPerToken   *float64 `json:"cache_creation_input_token_cost,omitempty"`
	CacheWrite1hCostPerToken *float64 `json:"cache_creation_input_token_cost_above_1hr,omitempty"`
}

// rateNames are the list keys of Rates; a tier's keys add a suffix, as in
// input_cost_per_token_above_200k_tokens.
var rateNames = []string{
	"input_cost_per_token",
	"output_cost_per_token",
	"cache_read_input_token_cost",
	"cache_creation_input_token_cost",
	"cache_creation_input_token_cost_above_1hr",
}

// Tier holds the rates of calls with more than AboveInputTokens input
// tokens.
type Tier struct {
	AboveInputTokens int `json:"above_input_tokens"`
	Rates
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
// output rates. Its long-input rates, keys such as
// input_cost_per_token_above_200k_tokens or a tiered_pricing table, become
// Tiers; a rate a tier lacks is the model's, or in a tiered_pricing table the
// tier's input rate, as LiteLLM prices them. Invalid models fail the load
// without changing the table.
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
		var entry listEntry
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
		var err error
		if len(entry.Tiered) > 0 {
			model.Pricing = entry.tieredPricing()
		} else if entry.Input != nil && entry.Output != nil {
			model.Pricing, err = flatPricing(data)
		}
		if err != nil {
			return nil, fmt.Errorf("catalog: decode model %q: %w", name, err)
		}
		if err := model.validate(name); err != nil {
			return nil, err
		}
		models[name] = model
	}
	return models, nil
}

type listEntry struct {
	Mode            string     `json:"mode"`
	Provider        string     `json:"litellm_provider"`
	MaxInputTokens  int        `json:"max_input_tokens"`
	MaxOutputTokens int        `json:"max_output_tokens"`
	Reasoning       *bool      `json:"supports_reasoning"`
	Input           *float64   `json:"input_cost_per_token"`
	Output          *float64   `json:"output_cost_per_token"`
	Tiered          []listTier `json:"tiered_pricing"`
}

// listTier prices calls whose input tokens are beyond Range's start and up
// to its end.
type listTier struct {
	Range      []float64 `json:"range"`
	Input      *float64  `json:"input_cost_per_token"`
	Output     *float64  `json:"output_cost_per_token"`
	CacheRead  *float64  `json:"cache_read_input_token_cost"`
	CacheWrite *float64  `json:"cache_creation_input_token_cost"`
}

// flatPricing returns the rates of an entry and the tiers its keys with an
// _above_<n>k_tokens suffix set, each starting from the entry's rates.
func flatPricing(data json.RawMessage) (*Pricing, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return nil, err
	}
	var pricing Pricing
	if err := json.Unmarshal(data, &pricing.Rates); err != nil {
		return nil, err
	}
	for key := range fields {
		m := tierKey.FindStringSubmatch(key)
		if m == nil {
			continue
		}
		thousands, err := strconv.Atoi(m[1])
		if err != nil {
			return nil, err
		}
		suffix := strings.TrimPrefix(key, rateNames[0])
		rates := make(map[string]json.RawMessage)
		for _, name := range rateNames {
			if raw, ok := fields[name+suffix]; ok {
				rates[name] = raw
			}
		}
		tier := Tier{AboveInputTokens: thousands * 1000, Rates: pricing.Rates.clone()}
		sub, _ := json.Marshal(rates)
		if err := json.Unmarshal(sub, &tier.Rates); err != nil {
			return nil, err
		}
		pricing.Tiers = append(pricing.Tiers, tier)
	}
	slices.SortFunc(pricing.Tiers, func(a, b Tier) int { return cmp.Compare(a.AboveInputTokens, b.AboveInputTokens) })
	return &pricing, nil
}

// tierKey is the input rate of a tier; rates for service tiers, batches
// and the like have further suffixes.
var tierKey = regexp.MustCompile(`^input_cost_per_token_above_(\d+)k_tokens$`)

// tieredPricing returns the rates of a tiered_pricing table, whose lowest
// range is the model's rates and each other one a tier beyond its start, or
// nil when a tier lacks a range, an input or an output rate. A tier without
// an output rate has the entry's.
func (e listEntry) tieredPricing() *Pricing {
	tiers := slices.Clone(e.Tiered)
	for _, t := range tiers {
		if len(t.Range) != 2 || t.Input == nil || cmp.Or(t.Output, e.Output) == nil {
			return nil
		}
	}
	slices.SortFunc(tiers, func(a, b listTier) int { return cmp.Compare(a.Range[0], b.Range[0]) })
	var pricing Pricing
	for i, t := range tiers {
		output := cmp.Or(t.Output, e.Output)
		rates := Rates{InputCostPerToken: *t.Input, OutputCostPerToken: *output, CacheReadCostPerToken: t.CacheRead, CacheWriteCostPerToken: t.CacheWrite}
		if i == 0 {
			pricing.Rates = rates
		} else {
			pricing.Tiers = append(pricing.Tiers, Tier{AboveInputTokens: int(t.Range[0]), Rates: rates})
		}
	}
	return &pricing
}

// Cost prices usage at the rates of the last tier its input tokens are
// above, or else at Rates. Cache reads and writes are priced at their rates
// and the rest of the input at the input rate; a cache count the vendor did
// not report is zero, its tokens priced as input. Usage without input tokens
// is an error: every call has some, so the vendor reported none. So are
// writes cached for an hour without their rate.
func (p Pricing) Cost(usage litellm.Usage) (Cost, error) {
	if err := p.validate(); err != nil {
		return Cost{}, fmt.Errorf("catalog: %w", err)
	}
	in, out, cacheRead, cacheWrite, cacheWrite1h := usage.InputTokens, usage.OutputTokens, usage.CacheReadTokens, usage.CacheWriteTokens, usage.CacheWrite1hTokens
	if in == 0 {
		return Cost{}, fmt.Errorf("catalog: the usage reports no input tokens")
	}
	if in < 0 || out < 0 || cacheRead < 0 || cacheWrite1h < 0 || cacheWrite1h > cacheWrite || cacheRead+cacheWrite > in {
		return Cost{}, fmt.Errorf("catalog: invalid token counts: cache reads and writes must fit within input tokens")
	}
	r := p.Rates
	for _, tier := range p.Tiers {
		if in > tier.AboveInputTokens {
			r = tier.Rates
		}
	}
	if cacheWrite1h > 0 && r.CacheWrite1hCostPerToken == nil {
		return Cost{}, fmt.Errorf("catalog: no rate for cache writes kept an hour")
	}
	cacheReadRate, cacheWriteRate := r.InputCostPerToken, r.InputCostPerToken
	if r.CacheReadCostPerToken != nil {
		cacheReadRate = *r.CacheReadCostPerToken
	}
	if r.CacheWriteCostPerToken != nil {
		cacheWriteRate = *r.CacheWriteCostPerToken
	}
	inputCost := float64(in-cacheRead-cacheWrite) * r.InputCostPerToken
	outputCost := float64(out) * r.OutputCostPerToken
	cacheReadCost := float64(cacheRead) * cacheReadRate
	cacheWriteCost := float64(cacheWrite-cacheWrite1h) * cacheWriteRate
	if cacheWrite1h > 0 {
		cacheWriteCost += float64(cacheWrite1h) * *r.CacheWrite1hCostPerToken
	}
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
	if err := p.Rates.validate(); err != nil {
		return err
	}
	for i, tier := range p.Tiers {
		if tier.AboveInputTokens <= 0 || i > 0 && tier.AboveInputTokens <= p.Tiers[i-1].AboveInputTokens {
			return fmt.Errorf("tiers must be above increasing, positive input token counts")
		}
		if err := tier.Rates.validate(); err != nil {
			return fmt.Errorf("tier above %d input tokens: %w", tier.AboveInputTokens, err)
		}
	}
	return nil
}

func (r Rates) validate() error {
	for _, rate := range []struct {
		name  string
		value *float64
	}{
		{"input", &r.InputCostPerToken}, {"output", &r.OutputCostPerToken},
		{"cache read", r.CacheReadCostPerToken}, {"cache write", r.CacheWriteCostPerToken},
		{"hour-long cache write", r.CacheWrite1hCostPerToken},
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
	pricing.Rates = pricing.Rates.clone()
	pricing.Tiers = slices.Clone(pricing.Tiers)
	for i := range pricing.Tiers {
		pricing.Tiers[i].Rates = pricing.Tiers[i].Rates.clone()
	}
	m.Pricing = &pricing
	return m
}

func (r Rates) clone() Rates {
	r.CacheReadCostPerToken = copyRate(r.CacheReadCostPerToken)
	r.CacheWriteCostPerToken = copyRate(r.CacheWriteCostPerToken)
	r.CacheWrite1hCostPerToken = copyRate(r.CacheWrite1hCostPerToken)
	return r
}

func copyRate(rate *float64) *float64 {
	if rate == nil {
		return nil
	}
	return new(*rate)
}
