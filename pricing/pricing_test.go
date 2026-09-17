package pricing

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

func TestRegistryCalculate(t *testing.T) {
	reg := NewRegistry()
	if err := reg.Set("model-a", ModelPricing{
		InputCostPerToken:      0.001,
		OutputCostPerToken:     0.002,
		CacheReadCostPerToken:  litellm.Float64Ptr(0.0005),
		CacheWriteCostPerToken: litellm.Float64Ptr(0.0015),
	}); err != nil {
		t.Fatalf("Set: %v", err)
	}

	cost, err := reg.Calculate("model-a", litellm.Usage{
		InputTokens:      litellm.IntPtr(100),
		OutputTokens:     litellm.IntPtr(20),
		CacheReadTokens:  litellm.IntPtr(40),
		CacheWriteTokens: litellm.IntPtr(10),
	})
	if err != nil {
		t.Fatalf("Calculate: %v", err)
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

func TestCalculateDoesNotLoadImplicitly(t *testing.T) {
	_, err := Calculate("model-a", litellm.Usage{InputTokens: litellm.IntPtr(1)}, nil)
	if err == nil || !strings.Contains(err.Error(), "not in table") {
		t.Fatalf("expected missing table error, got %v", err)
	}
}

func TestRegistryLoadFromReader(t *testing.T) {
	reg := NewRegistry()
	err := reg.LoadFromReader(strings.NewReader(`{
		"sample_spec": {},
		"model-a": {
			"input_cost_per_token": 0.001,
			"output_cost_per_token": 0.002,
			"cache_read_input_token_cost": 0.0005,
			"litellm_provider": "openai",
			"max_input_tokens": 128000,
			"max_output_tokens": 4096,
			"supports_function_calling": true,
			"supports_vision": true,
			"supports_reasoning": true
		}
	}`))
	if err != nil {
		t.Fatalf("LoadFromReader: %v", err)
	}
	price, ok := reg.Get("openai/model-a")
	if !ok || price.InputCostPerToken != 0.001 || price.OutputCostPerToken != 0.002 {
		t.Fatalf("price = %+v, ok=%v", price, ok)
	}
	caps, ok := reg.Capabilities("model-a")
	if !ok || caps.Provider != "openai" || !caps.SupportsTools || !caps.SupportsVision || !caps.SupportsReasoning {
		t.Fatalf("capabilities = %+v, ok=%v", caps, ok)
	}
}

func TestRegistryLoadFromURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pricing.json" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"model-a":{"input_cost_per_token":0.001,"output_cost_per_token":0.002}}`))
	}))
	defer server.Close()

	reg := NewRegistry()
	if err := reg.LoadFromURL(context.Background(), server.URL+"/pricing.json"); err != nil {
		t.Fatalf("LoadFromURL: %v", err)
	}
	if _, ok := reg.Get("model-a"); !ok {
		t.Fatalf("model-a pricing not loaded")
	}
}

func TestCalculateUnknownAndInvalidUsage(t *testing.T) {
	table := map[string]ModelPricing{"m": {InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadCostPerToken: litellm.Float64Ptr(0.5)}}
	for _, usage := range []litellm.Usage{
		{},
		{InputTokens: litellm.IntPtr(10), OutputTokens: litellm.IntPtr(1)},
		{InputTokens: litellm.IntPtr(10), OutputTokens: litellm.IntPtr(1), CacheReadTokens: litellm.IntPtr(8), CacheWriteTokens: litellm.IntPtr(3)},
		{InputTokens: litellm.IntPtr(-1), OutputTokens: litellm.IntPtr(1), CacheReadTokens: litellm.IntPtr(0)},
	} {
		if _, err := Calculate("m", usage, table); err == nil {
			t.Fatalf("expected error for %+v", usage)
		}
	}
	zero := litellm.Usage{InputTokens: litellm.IntPtr(0), OutputTokens: litellm.IntPtr(0), CacheReadTokens: litellm.IntPtr(0)}
	if cost, err := Calculate("m", zero, table); err != nil || cost.Total != 0 {
		t.Fatalf("known zero: %+v %v", cost, err)
	}
	table["m"] = ModelPricing{InputCostPerToken: 1, OutputCostPerToken: 2}
	if cost, err := Calculate("m", litellm.Usage{InputTokens: litellm.IntPtr(10), OutputTokens: litellm.IntPtr(2)}, table); err != nil || cost.Total != 14 {
		t.Fatalf("equal cache rates: %+v %v", cost, err)
	}
}

func TestFreeCacheRatesAndRegistryOwnership(t *testing.T) {
	r := NewRegistry()
	zero := 0.0
	price := ModelPricing{InputCostPerToken: 1, OutputCostPerToken: 2, CacheReadCostPerToken: &zero, CacheWriteCostPerToken: &zero}
	if err := r.Set("free-cache", price); err != nil {
		t.Fatal(err)
	}
	zero = 99
	got, ok := r.Get("free-cache")
	if !ok || got.CacheReadCostPerToken == nil || *got.CacheReadCostPerToken != 0 {
		t.Fatalf("price = %+v", got)
	}
	*got.CacheReadCostPerToken = 100
	usage := litellm.Usage{InputTokens: litellm.IntPtr(10), OutputTokens: litellm.IntPtr(2), CacheReadTokens: litellm.IntPtr(6), CacheWriteTokens: litellm.IntPtr(4)}
	if cost, err := r.Calculate("free-cache", usage); err != nil || cost.Total != 4 || cost.CacheRead != 0 || cost.CacheWrite != 0 {
		t.Fatalf("free cache cost = %+v, %v", cost, err)
	}
	if err := r.LoadFromReader(strings.NewReader(`{"free":{"input_cost_per_token":1,"output_cost_per_token":2,"cache_read_input_token_cost":0,"cache_creation_input_token_cost":0},"inherited":{"input_cost_per_token":1,"output_cost_per_token":2}}`)); err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]float64{"free": 4, "inherited": 14} {
		cost, err := r.Calculate(model, usage)
		if err != nil || cost.Total != want {
			t.Fatalf("%s: %+v %v", model, cost, err)
		}
	}
	for _, value := range []float64{-1, math.NaN(), math.Inf(1)} {
		if err := r.Set("bad", ModelPricing{CacheReadCostPerToken: &value}); err == nil {
			t.Fatal("accepted invalid rate")
		}
	}
}
