package openai

import (
	"testing"

	"github.com/voocel/litellm"
)

func TestCapabilitiesReasoningModel(t *testing.T) {
	provider := mustProvider(t)
	caps := provider.Capabilities("gpt-5.1")
	if caps.Provider != "openai" || caps.Model != "gpt-5.1" {
		t.Fatalf("caps = %+v", caps)
	}
	if caps.Thinking.Supported != litellm.SupportPartial || caps.Thinking.Disable != litellm.SupportYes {
		t.Fatalf("thinking caps = %+v", caps.Thinking)
	}
	if !caps.Thinking.SupportsEffort("xhigh") || caps.Thinking.SupportsEffort("minimal") {
		t.Fatalf("thinking caps = %+v", caps.Thinking)
	}
	if caps.Streaming.NativeResponses != litellm.SupportYes {
		t.Fatalf("native responses = %v, want yes", caps.Streaming.NativeResponses)
	}
}

func TestCapabilitiesNonReasoningModel(t *testing.T) {
	provider := mustProvider(t)
	caps := provider.Capabilities("gpt-4.1")
	if caps.Thinking.Supported != litellm.SupportNo || len(caps.Thinking.Efforts) != 0 {
		t.Fatalf("thinking caps = %+v", caps.Thinking)
	}
}

// TestCapabilitiesStructuredOutputsByEndpoint verifies that Structured
// Outputs are an official endpoint contract, while compatible custom
// endpoints remain Unknown unless the caller explicitly declares support.
func TestCapabilitiesStructuredOutputsByEndpoint(t *testing.T) {
	official := Config{APIKey: "test"}
	custom := Config{APIKey: "test", BaseURL: "https://relay.example.com/v1"}
	cases := []struct {
		name  string
		cfg   Config
		model string
		want  litellm.Support
	}{
		{"official endpoint", official, "gpt-5.6", litellm.SupportYes},
		{"custom endpoint", custom, "gpt-5.6", litellm.SupportUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider, err := New(tc.cfg)
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}
			caps := provider.Capabilities(tc.model)
			if caps.Structured.JSONSchema != tc.want || caps.Structured.Strict != tc.want {
				t.Fatalf("structured caps = %+v, want json_schema/strict %v", caps.Structured, tc.want)
			}
		})
	}
	provider, err := New(custom)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if got := provider.Capabilities("gpt-4o").Structured.JSONObject; got != litellm.SupportUnknown {
		t.Fatalf("custom base URL json_object = %v, want unknown", got)
	}
}

// TestCapabilitiesPromptCacheParamsGating verifies that prompt cache params
// are only advertised for the official endpoint by default: compatible
// backends have no unknown-field contract (strict ones return 400/422), so
// custom BaseURLs degrade to SupportUnknown unless explicitly opted in.
func TestCapabilitiesPromptCacheParamsGating(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want litellm.Support
	}{
		{"default official endpoint", Config{APIKey: "test"}, litellm.SupportYes},
		{"explicit official endpoint", Config{APIKey: "test", BaseURL: "https://api.openai.com/v1"}, litellm.SupportYes},
		{"third-party compatible endpoint", Config{APIKey: "test", BaseURL: "https://relay.example.com/v1"}, litellm.SupportUnknown},
		{"third-party with opt-in", Config{APIKey: "test", BaseURL: "https://relay.example.com/v1", PromptCacheParams: true}, litellm.SupportYes},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider, err := New(tc.cfg)
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}
			caps := provider.Capabilities("gpt-4.1")
			if caps.Cache.PromptKey != tc.want || caps.Cache.Retention != tc.want {
				t.Fatalf("prompt cache caps = (key=%v retention=%v), want %v", caps.Cache.PromptKey, caps.Cache.Retention, tc.want)
			}
		})
	}
}
