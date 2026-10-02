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
	price := Pricing{
		InputCostPerToken:      0.001,
		OutputCostPerToken:     0.002,
		CacheReadCostPerToken:  new(0.0005),
		CacheWriteCostPerToken: new(0.0015),
	}
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
		Pricing:         &Pricing{InputCostPerToken: 0.001, OutputCostPerToken: 0.002, CacheReadCostPerToken: new(0.0005)},
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
		{"model-a", &Pricing{InputCostPerToken: 1, OutputCostPerToken: 2}},
		{"site-a/model-a", &Pricing{InputCostPerToken: 3, OutputCostPerToken: 4}},
		{"site-b/model-a", &Pricing{InputCostPerToken: 5, OutputCostPerToken: 6}},
		{"site-a/model-b", &Pricing{InputCostPerToken: 7, OutputCostPerToken: 8}},
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
		{"input rate", Model{Pricing: &Pricing{InputCostPerToken: -1}}, `{"bad":{"mode":"chat","input_cost_per_token":-1,"output_cost_per_token":0}}`},
		{"output rate", Model{Pricing: &Pricing{OutputCostPerToken: -1}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":-1}}`},
		{"cache read rate", Model{Pricing: &Pricing{CacheReadCostPerToken: new(-1.0)}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":0,"cache_read_input_token_cost":-1}}`},
		{"cache write rate", Model{Pricing: &Pricing{CacheWriteCostPerToken: new(-1.0)}}, `{"bad":{"mode":"chat","input_cost_per_token":0,"output_cost_per_token":0,"cache_creation_input_token_cost":-1}}`},
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
	price := Pricing{InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadCostPerToken: new(0.5)}
	for _, usage := range []litellm.Usage{
		{InputTokens: 10, OutputTokens: 1, CacheReadTokens: 8, CacheWriteTokens: 3},
		{InputTokens: -1, OutputTokens: 1},
	} {
		if _, err := price.Cost(usage); err == nil {
			t.Fatalf("expected error for %+v", usage)
		}
	}
	if cost, err := price.Cost(litellm.Usage{}); err != nil || cost.Total != 0 {
		t.Fatalf("zero usage: %+v %v", cost, err)
	}
	// A vendor that reports no cache counts, such as MiniMax, is priced as
	// uncached input.
	if cost, err := price.Cost(litellm.Usage{InputTokens: 10, OutputTokens: 1}); err != nil || cost.Total != 12 {
		t.Fatalf("unreported cache: %+v %v", cost, err)
	}
	price = Pricing{InputCostPerToken: 1, OutputCostPerToken: 2}
	if cost, err := price.Cost(litellm.Usage{InputTokens: 10, OutputTokens: 2}); err != nil || cost.Total != 14 {
		t.Fatalf("equal cache rates: %+v %v", cost, err)
	}
}

func TestFreeCacheRatesAndOwnership(t *testing.T) {
	var c Catalog
	zero := 0.0
	price := Pricing{InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadCostPerToken: &zero, CacheWriteCostPerToken: &zero}
	if err := c.Set("free-cache", Model{Pricing: &price}); err != nil {
		t.Fatal(err)
	}
	zero = 99
	got, ok := c.Get("free-cache")
	if !ok || got.Pricing.CacheReadCostPerToken == nil || *got.Pricing.CacheReadCostPerToken != 0 {
		t.Fatalf("model = %+v", got)
	}
	*got.Pricing.CacheReadCostPerToken = 100
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
		if err := c.Set("bad", Model{Pricing: &Pricing{CacheReadCostPerToken: &value}}); err == nil {
			t.Fatal("accepted invalid rate")
		}
	}
	if err := c.Set("bad", Model{MaxOutputTokens: -1}); err == nil {
		t.Fatal("accepted a negative limit")
	}
}
