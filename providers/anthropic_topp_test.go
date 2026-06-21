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

func TestAnthropicBuildHTTPRequest_SetsConfiguredUserAgent(t *testing.T) {
	p := NewAnthropic(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"user_agent": "custom-client/1.0",
		},
	})
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}

	httpReq, err := p.buildHTTPRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest: %v", err)
	}

	if got := httpReq.Header.Get("User-Agent"); got != "custom-client/1.0" {
		t.Fatalf("User-Agent = %q, want custom-client/1.0", got)
	}
}

func TestAnthropicBuildHTTPRequest_SetsExtraHeaders(t *testing.T) {
	p := NewAnthropic(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"headers": map[string]string{
				"X-App":               "claude-code",
				"X-Stainless-Runtime": "node",
			},
		},
	})
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}

	httpReq, err := p.buildHTTPRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest: %v", err)
	}

	if got := httpReq.Header.Get("X-App"); got != "claude-code" {
		t.Fatalf("X-App = %q, want claude-code", got)
	}
	if got := httpReq.Header.Get("X-Stainless-Runtime"); got != "node" {
		t.Fatalf("X-Stainless-Runtime = %q, want node", got)
	}
}

func TestAnthropicBuildHTTPRequest_SetsAnthropicBetaHeader(t *testing.T) {
	p := NewAnthropic(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"anthropic_beta": []string{"beta-one", "beta-two"},
		},
	})
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}

	httpReq, err := p.buildHTTPRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("buildHTTPRequest: %v", err)
	}

	if got := httpReq.Header.Get("anthropic-beta"); got != "beta-one,beta-two" {
		t.Fatalf("anthropic-beta = %q, want beta-one,beta-two", got)
	}
}

func TestAnthropicBuildHTTPRequest_RejectsInvalidAnthropicBeta(t *testing.T) {
	p := NewAnthropic(ProviderConfig{
		APIKey: "test-key",
		Extra: map[string]any{
			"anthropic_beta": []any{"beta-one", 42},
		},
	})
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}

	if _, err := p.buildHTTPRequest(context.Background(), req, false); err == nil {
		t.Fatal("buildHTTPRequest error = nil, want invalid anthropic_beta error")
	}
}

func TestAnthropicBuildHTTPRequest_SetsMetadataUserID(t *testing.T) {
	p := NewAnthropic(ProviderConfig{APIKey: "test-key"})
	req := &Request{
		Model:    "claude-sonnet-4-6",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Extra: map[string]any{
			"metadata_user_id": "claude-code-user",
		},
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

	metadata, ok := got["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("metadata missing or invalid: %#v", got["metadata"])
	}
	if metadata["user_id"] != "claude-code-user" {
		t.Fatalf("metadata.user_id = %#v, want claude-code-user", metadata["user_id"])
	}
}
