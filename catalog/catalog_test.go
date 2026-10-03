package catalog

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestPricingCost(t *testing.T) {
	price := Pricing{Rates: Rates{
		Input:      0.001,
		Output:     0.002,
		CacheRead:  new(0.0005),
		CacheWrite: new(0.0015),
	}}
	cost, err := price.Cost(litellm.Usage{
		InputTokens:      100,
		OutputTokens:     20,
		CacheReadTokens:  40,
		CacheWriteTokens: 10,
	})
	if err != nil {
		t.Fatalf("Cost: %v", err)
	}
	if !close(cost.Input, 0.05) || !close(cost.Output, 0.04) || !close(cost.CacheRead, 0.02) || !close(cost.CacheWrite, 0.015) {
		t.Fatalf("cost = %+v", cost)
	}
	if !close(cost.Total, 0.125) {
		t.Fatalf("total = %v", cost.Total)
	}
}

func close(a, b float64) bool {
	return math.Abs(a-b) < 1e-12
}

func TestCatalogDoesNotLoadImplicitly(t *testing.T) {
	var c Catalog
	if _, ok := c.Get("gpt-5"); ok {
		t.Fatal("empty catalog found a model")
	}
}

// Chat models keep their limits, reasoning flag and prices; other modes and
// the format's sample entry are skipped.
func TestLoadFromReader(t *testing.T) {
	var c Catalog
	err := c.LoadFromReader(strings.NewReader(`{
		"sample_spec": {"max_input_tokens": "max input tokens"},
		"model-a": {
			"mode": "chat",
			"litellm_provider": "openai",
			"input_cost_per_token": 0.001,
			"output_cost_per_token": 0.002,
			"cache_read_input_token_cost": 0.0005,
			"max_input_tokens": 128000,
			"max_output_tokens": 4096,
			"supports_reasoning": true
		},
		"model-b": {"mode": "responses", "litellm_provider": "openai", "max_output_tokens": 100},
		"embed": {"mode": "embedding", "input_cost_per_token": 0.001, "output_cost_per_token": 0}
	}`))
	if err != nil {
		t.Fatalf("LoadFromReader: %v", err)
	}
	want := Model{
		Provider:        "openai",
		MaxInputTokens:  128000,
		MaxOutputTokens: 4096,
		Reasoning:       new(true),
		Pricing:         &Pricing{Rates: Rates{Input: 0.001, Output: 0.002, CacheRead: new(0.0005)}},
	}
	if got, ok := c.Get("model-a"); !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("model-a = %+v, ok=%v", got, ok)
	}
	if got, ok := c.Get("model-b"); !ok || got.Pricing != nil || got.MaxOutputTokens != 100 {
		t.Fatalf("model-b = %+v, ok=%v", got, ok)
	}
	var names []string
	for name := range c.All() {
		names = append(names, name)
	}
	if !slices.Equal(names, []string{"model-a", "model-b"}) {
		t.Fatalf("names = %v", names)
	}
}

func TestLoadFromURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models.json" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"model-a":{"mode":"chat","input_cost_per_token":0.001,"output_cost_per_token":0.002}}`))
	}))
	defer server.Close()

	var c Catalog
	if err := c.LoadFromURL(context.Background(), server.URL+"/models.json"); err != nil {
		t.Fatalf("LoadFromURL: %v", err)
	}
	if _, ok := c.Get("model-a"); !ok {
		t.Fatalf("model-a not loaded")
	}
}

func TestGetKeepsPricingSourcesSeparate(t *testing.T) {
	var c Catalog
	if err := c.LoadFromReader(strings.NewReader(`{
		"model-a":{"mode":"chat","input_cost_per_token":1,"output_cost_per_token":2},
		"site-a/model-a":{"mode":"chat","input_cost_per_token":3,"output_cost_per_token":4},
		"site-b/model-a":{"mode":"chat","input_cost_per_token":5,"output_cost_per_token":6},
		"site-a/model-b":{"mode":"chat","input_cost_per_token":7,"output_cost_per_token":8}
	}`)); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		want *Pricing
	}{
		{"model-a", &Pricing{Rates: Rates{Input: 1, Output: 2}}},
		{"site-a/model-a", &Pricing{Rates: Rates{Input: 3, Output: 4}}},
		{"site-b/model-a", &Pricing{Rates: Rates{Input: 5, Output: 6}}},
		{"site-a/model-b", &Pricing{Rates: Rates{Input: 7, Output: 8}}},
		{"unknown/model-a", nil},
		{"site-b/model-b", nil},
		{"model-b", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := c.Get(tt.name)
			if ok != (tt.want != nil) || !reflect.DeepEqual(got, Model{Pricing: tt.want}) {
				t.Fatalf("Get(%q) = %+v, %v; want pricing %+v", tt.name, got, ok, tt.want)
			}
		})
	}
}

func TestLoadRejectsInvalidModelsWithoutReplacingCatalog(t *testing.T) {
	for _, tt := range []struct {
		name  string
		model Model
		data  string
	}{
		{"empty name", Model{}, `{" ":{"mode":"chat"}}`},
		{"input limit", Model{MaxInputTokens: -1}, `{"bad":{"mode":"chat","max_input_tokens":-1}}`},
		{"output limit", Model{MaxOutputTokens: -1}, `{"bad":{"mode":"responses","max_output_tokens":-1}}`},
		{"input rate", Model{Pricing: &Pricing{Rates: Rates{Input: -1}}}, `{"bad":{"mode":"chat","input_cost_per_token":-1,"output_cost_per_token":0}}`},
		{"output rate", Model{Pricing: &Pricing{Rates: Rates{Output: -1}}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":-1}}`},
		{"cache read rate", Model{Pricing: &Pricing{Rates: Rates{CacheRead: new(-1.0)}}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":0,"cache_read_input_token_cost":-1}}`},
		{"cache write rate", Model{Pricing: &Pricing{Rates: Rates{CacheWrite: new(-1.0)}}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":0,"cache_creation_input_token_cost":-1}}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var c Catalog
			original := Model{MaxInputTokens: 100, Reasoning: new(false)}
			if err := c.Set("original", original); err != nil {
				t.Fatal(err)
			}
			name := "bad"
			if tt.name == "empty name" {
				name = " "
			}
			setErr := c.Set(name, tt.model)
			loadErr := c.LoadFromReader(strings.NewReader(tt.data))
			if setErr == nil || loadErr == nil {
				t.Fatalf("invalid model accepted: Set=%v Load=%v", setErr, loadErr)
			}
			if setErr.Error() != loadErr.Error() {
				t.Fatalf("inconsistent validation: Set=%v Load=%v", setErr, loadErr)
			}
			if name == "bad" && !strings.Contains(loadErr.Error(), `"bad"`) {
				t.Fatalf("error has no model name: %v", loadErr)
			}
			entries := make(map[string]Model)
			for name, model := range c.All() {
				entries[name] = model
			}
			if !reflect.DeepEqual(entries, map[string]Model{"original": original}) {
				t.Fatalf("failed load changed catalog: %+v", entries)
			}
		})
	}
}

func TestReasoningFacts(t *testing.T) {
	var c Catalog
	if err := c.LoadFromReader(strings.NewReader(`{
		"unknown":{"mode":"chat"},
		"unsupported":{"mode":"chat","supports_reasoning":false},
		"supported":{"mode":"responses","supports_reasoning":true}
	}`)); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		want *bool
	}{
		{"unknown", nil},
		{"unsupported", new(false)},
		{"supported", new(true)},
	} {
		model, ok := c.Get(tt.name)
		if !ok || !reflect.DeepEqual(model.Reasoning, tt.want) {
			t.Fatalf("%s: reasoning=%v, want %v", tt.name, model.Reasoning, tt.want)
		}
		data, err := json.Marshal(model)
		if err != nil {
			t.Fatal(err)
		}
		var decoded Model
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(decoded.Reasoning, tt.want) {
			t.Fatalf("%s: JSON lost reasoning fact: %s", tt.name, data)
		}
	}
}

func TestReasoningOwnership(t *testing.T) {
	var c Catalog
	supported := true
	if err := c.Set("model", Model{Reasoning: &supported}); err != nil {
		t.Fatal(err)
	}
	supported = false
	model, _ := c.Get("model")
	if model.Reasoning == nil || !*model.Reasoning {
		t.Fatal("Set retained the caller's reasoning pointer")
	}
	*model.Reasoning = false
	for _, model := range c.All() {
		if model.Reasoning == nil || !*model.Reasoning {
			t.Fatal("Get exposed the catalog's reasoning pointer")
		}
		*model.Reasoning = false
	}
	model, _ = c.Get("model")
	if model.Reasoning == nil || !*model.Reasoning {
		t.Fatal("All exposed the catalog's reasoning pointer")
	}
}

func TestCostUnreportedCacheAndInvalidUsage(t *testing.T) {
	price := Pricing{Rates: Rates{Input: 1, Output: 2, CacheRead: new(0.5)}}
	for _, usage := range []litellm.Usage{
		{InputTokens: 10, OutputTokens: 1, CacheReadTokens: 8, CacheWriteTokens: 3},
		{InputTokens: -1, OutputTokens: 1},
	} {
		if _, err := price.Cost(usage); err == nil {
			t.Fatalf("expected error for %+v", usage)
		}
	}
	// Every call has input tokens: usage without any was not reported.
	if _, err := price.Cost(litellm.Usage{OutputTokens: 3}); err == nil {
		t.Fatal("priced usage the vendor did not report")
	}
	// A vendor that reports no cache counts, such as MiniMax, is priced as
	// uncached input.
	if cost, err := price.Cost(litellm.Usage{InputTokens: 10, OutputTokens: 1}); err != nil || cost.Total != 12 {
		t.Fatalf("unreported cache: %+v %v", cost, err)
	}
	price = Pricing{Rates: Rates{Input: 1, Output: 2}}
	if cost, err := price.Cost(litellm.Usage{InputTokens: 10, OutputTokens: 2}); err != nil || cost.Total != 14 {
		t.Fatalf("equal cache rates: %+v %v", cost, err)
	}
}

// Writes cached for an hour are priced at their own rate, and only at it.
func TestCostOfHourLongCacheWrites(t *testing.T) {
	price := Pricing{Rates: Rates{Input: 1, Output: 2, CacheWrite: new(1.25), CacheWrite1h: new(2.0)}}
	usage := litellm.Usage{InputTokens: 10, OutputTokens: 1, CacheWriteTokens: 8, CacheWrite1hTokens: 4}
	if cost, err := price.Cost(usage); err != nil || cost.CacheWrite != 4*1.25+4*2 || cost.Total != 2+2+13 {
		t.Fatalf("cost = %+v, %v", cost, err)
	}
	price.CacheWrite1h = nil
	if _, err := price.Cost(usage); err == nil {
		t.Fatal("priced hour-long writes without their rate")
	}
	if _, err := price.Cost(litellm.Usage{InputTokens: 10, CacheWriteTokens: 2, CacheWrite1hTokens: 3}); err == nil {
		t.Fatal("accepted more hour-long writes than writes")
	}

	var c Catalog
	if err := c.LoadFromReader(strings.NewReader(`{"m":{"mode":"chat","input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost_above_1hr":2}}`)); err != nil {
		t.Fatal(err)
	}
	model, _ := c.Get("m")
	if rate := model.Pricing.CacheWrite1h; rate == nil || *rate != 2 {
		t.Fatalf("hour-long write rate = %v", rate)
	}
	*model.Pricing.CacheWrite1h = 9
	if again, _ := c.Get("m"); *again.Pricing.CacheWrite1h != 2 {
		t.Fatal("Get exposed the catalog's hour-long write rate")
	}
}

func TestFreeCacheRatesAndOwnership(t *testing.T) {
	var c Catalog
	zero := 0.0
	price := Pricing{Rates: Rates{Input: 1, Output: 2, CacheRead: &zero, CacheWrite: &zero}}
	if err := c.Set("free-cache", Model{Pricing: &price}); err != nil {
		t.Fatal(err)
	}
	zero = 99
	got, ok := c.Get("free-cache")
	if !ok || got.Pricing.CacheRead == nil || *got.Pricing.CacheRead != 0 {
		t.Fatalf("model = %+v", got)
	}
	*got.Pricing.CacheRead = 100
	usage := litellm.Usage{InputTokens: 10, OutputTokens: 2, CacheReadTokens: 6, CacheWriteTokens: 4}
	stored, _ := c.Get("free-cache")
	if cost, err := stored.Pricing.Cost(usage); err != nil || cost.Total != 4 || cost.CacheRead != 0 || cost.CacheWrite != 0 {
		t.Fatalf("free cache cost = %+v, %v", cost, err)
	}
	if err := c.LoadFromReader(strings.NewReader(`{"free":{"mode":"chat","input_cost_per_token":1,"output_cost_per_token":2,"cache_read_input_token_cost":0,"cache_creation_input_token_cost":0},"inherited":{"mode":"chat","input_cost_per_token":1,"output_cost_per_token":2}}`)); err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]float64{"free": 4, "inherited": 14} {
		model, _ := c.Get(name)
		cost, err := model.Pricing.Cost(usage)
		if err != nil || cost.Total != want {
			t.Fatalf("%s: %+v %v", name, cost, err)
		}
	}
	for _, value := range []float64{-1, math.NaN(), math.Inf(1)} {
		if err := c.Set("bad", Model{Pricing: &Pricing{Rates: Rates{CacheRead: &value}}}); err == nil {
			t.Fatal("accepted invalid rate")
		}
	}
	if err := c.Set("bad", Model{MaxOutputTokens: -1}); err == nil {
		t.Fatal("accepted a negative limit")
	}
}

// Long-input rates load as tiers, in both of the list's forms, with the
// rates a tier lacks filled in as LiteLLM prices them.
func TestLoadTiers(t *testing.T) {
	var c Catalog
	err := c.LoadFromReader(strings.NewReader(`{
		"claude": {
			"mode": "chat",
			"input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6,
			"cache_read_input_token_cost": 3e-7, "cache_creation_input_token_cost": 3.75e-6,
			"cache_creation_input_token_cost_above_1hr": 6e-6,
			"input_cost_per_token_above_200k_tokens": 6e-6, "output_cost_per_token_above_200k_tokens": 22.5e-6,
			"cache_read_input_token_cost_above_200k_tokens": 6e-7, "cache_creation_input_token_cost_above_200k_tokens": 7.5e-6,
			"cache_creation_input_token_cost_above_1hr_above_200k_tokens": 12e-6,
			"input_cost_per_token_above_200k_tokens_priority": 1
		},
		"coder": {
			"mode": "chat",
			"input_cost_per_token": 1, "output_cost_per_token": 2,
			"input_cost_per_token_above_128k_tokens": 5, "output_cost_per_token_above_128k_tokens": 6,
			"input_cost_per_token_above_32k_tokens": 3
		},
		"flash": {
			"mode": "chat",
			"tiered_pricing": [
				{"input_cost_per_token": 4, "range": [256000, 1000000]},
				{"input_cost_per_token": 1, "output_cost_per_token": 2, "cache_read_input_token_cost": 0.5, "range": [0, 256000]}
			],
			"output_cost_per_token": 8
		},
		"no output": {"mode": "chat", "tiered_pricing": [{"input_cost_per_token": 1, "range": [0, 1000]}]},
		"empty table": {"mode": "chat", "tiered_pricing": [], "input_cost_per_token": 1, "output_cost_per_token": 2}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]*Pricing{
		"claude": {
			Rates: Rates{Input: 3e-6, Output: 15e-6, CacheRead: new(3e-7), CacheWrite: new(3.75e-6), CacheWrite1h: new(6e-6)},
			Tiers: []Tier{{AboveInputTokens: 200000, Rates: Rates{Input: 6e-6, Output: 22.5e-6, CacheRead: new(6e-7), CacheWrite: new(7.5e-6), CacheWrite1h: new(12e-6)}}},
		},
		"coder": {
			Rates: Rates{Input: 1, Output: 2},
			Tiers: []Tier{
				{AboveInputTokens: 32000, Rates: Rates{Input: 3, Output: 2}},
				{AboveInputTokens: 128000, Rates: Rates{Input: 5, Output: 6}},
			},
		},
		"flash": {
			Rates: Rates{Input: 1, Output: 2, CacheRead: new(0.5)},
			Tiers: []Tier{{AboveInputTokens: 256000, Rates: Rates{Input: 4, Output: 8}}},
		},
		"no output":   nil,
		"empty table": {Rates: Rates{Input: 1, Output: 2}},
	} {
		if got, _ := c.Get(name); !reflect.DeepEqual(got.Pricing, want) {
			t.Errorf("%s: pricing = %+v, want %+v", name, got.Pricing, want)
		}
	}
}

// Every token of a call above a tier's input count is priced at its rates.
func TestPricingCostTiers(t *testing.T) {
	price := Pricing{
		Rates: Rates{Input: 1, Output: 2},
		Tiers: []Tier{
			{AboveInputTokens: 100, Rates: Rates{Input: 3, Output: 4, CacheWrite1h: new(5.0)}},
			{AboveInputTokens: 200, Rates: Rates{Input: 10, Output: 20, CacheRead: new(1.0)}},
		},
	}
	for _, tc := range []struct {
		usage litellm.Usage
		want  float64
	}{
		{litellm.Usage{InputTokens: 100, OutputTokens: 10}, 100 + 20},
		{litellm.Usage{InputTokens: 101, OutputTokens: 10, CacheWriteTokens: 1, CacheWrite1hTokens: 1}, 100*3 + 40 + 5},
		{litellm.Usage{InputTokens: 300, OutputTokens: 10, CacheReadTokens: 100}, 200*10 + 200 + 100},
	} {
		if cost, err := price.Cost(tc.usage); err != nil || cost.Total != tc.want {
			t.Errorf("%+v: cost = %+v, %v; want %v", tc.usage, cost, err, tc.want)
		}
	}
	if _, err := price.Cost(litellm.Usage{InputTokens: 300, CacheWriteTokens: 1, CacheWrite1hTokens: 1}); err == nil {
		t.Error("priced hour-long writes the tier has no rate for")
	}
}

func TestTiersAreValidatedAndOwned(t *testing.T) {
	var c Catalog
	for _, tiers := range [][]Tier{
		{{AboveInputTokens: 0}},
		{{AboveInputTokens: 200}, {AboveInputTokens: 100}},
		{{AboveInputTokens: 100, Rates: Rates{Input: -1}}},
	} {
		if err := c.Set("bad", Model{Pricing: &Pricing{Tiers: tiers}}); err == nil {
			t.Errorf("accepted tiers %+v", tiers)
		}
	}
	rate := 1.0
	if err := c.Set("m", Model{Pricing: &Pricing{Tiers: []Tier{{AboveInputTokens: 100, Rates: Rates{CacheRead: &rate}}}}}); err != nil {
		t.Fatal(err)
	}
	rate = 2
	got, _ := c.Get("m")
	*got.Pricing.Tiers[0].CacheRead = 3
	got.Pricing.Tiers[0].Input = 3
	if again, _ := c.Get("m"); *again.Pricing.Tiers[0].CacheRead != 1 || again.Pricing.Tiers[0].Input != 0 {
		t.Fatalf("tier = %+v", again.Pricing.Tiers[0])
	}
}
