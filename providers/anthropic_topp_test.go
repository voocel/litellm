package providers

import (
	"context"
	"encoding/json"
	"io"
	"testing"
)

// Claude models reject a request that specifies both temperature and top_p
// ("`temperature` and `top_p` cannot both be specified for this model").
// The client backfills top_p by default, so when a caller (or a default) sets
// temperature, the Anthropic provider must not also send top_p.
func TestAnthropicBuildHTTPRequest_DropsTopPWhenTemperatureSet(t *testing.T) {
	p := NewAnthropic(ProviderConfig{APIKey: "test-key"})
	temp, topP := 0.7, 0.9
	req := &Request{
		Model:       "claude-sonnet-4-6",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: &temp,
		TopP:        &topP,
	}

	httpReq, err := p.buildHTTPRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest: %v", err)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if _, ok := got["top_p"]; ok {
		t.Errorf("top_p must be omitted when temperature is set; body=%s", body)
	}
	if _, ok := got["temperature"]; !ok {
		t.Errorf("temperature should be present; body=%s", body)
	}
}

// When only top_p is set, it must still be sent (temperature is the one dropped
// only when both are present).
func TestAnthropicBuildHTTPRequest_KeepsTopPWhenTemperatureUnset(t *testing.T) {
	p := NewAnthropic(ProviderConfig{APIKey: "test-key"})
	topP := 0.9
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
		TopP:     &topP,
	}

	httpReq, err := p.buildHTTPRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest: %v", err)
	}
	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	if _, ok := got["top_p"]; !ok {
		t.Errorf("top_p should be present when temperature is unset; body=%s", body)
	}
}
