package litellm

import (
	"context"
	"strings"
	"testing"
	"time"
)

type stubProvider struct {
	name string
}

func (p *stubProvider) Name() string                                           { return p.name }
func (p *stubProvider) Validate() error                                        { return nil }
func (p *stubProvider) Chat(context.Context, *Request) (*Response, error)      { return &Response{}, nil }
func (p *stubProvider) Stream(context.Context, *Request) (StreamReader, error) { return nil, nil }

func TestCustomProviderRegistrationAndCreation(t *testing.T) {
	providerName := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_"))
	var captured ProviderConfig

	err := RegisterProvider(providerName, func(cfg ProviderConfig) Provider {
		captured = cfg
		return &stubProvider{name: providerName}
	})
	if err != nil {
		t.Fatalf("RegisterProvider returned error: %v", err)
	}

	client, err := NewWithProvider(providerName, ProviderConfig{
		Timeout: 15 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewWithProvider returned error: %v", err)
	}
	if client.ProviderName() != providerName {
		t.Fatalf("provider name = %q, want %q", client.ProviderName(), providerName)
	}
	if captured.Resilience.RequestTimeout != 15*time.Second {
		t.Fatalf("request timeout = %v, want %v", captured.Resilience.RequestTimeout, 15*time.Second)
	}
	if captured.HTTPClient == nil {
		t.Fatal("expected HTTPClient to be injected for custom provider")
	}
}

func TestRegisterProviderWithDescriptorAppliesDefaultURL(t *testing.T) {
	providerName := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_"))
	var captured ProviderConfig

	err := RegisterProviderWithDescriptor(ProviderDescriptor{
		Name:       providerName,
		DefaultURL: "https://example.test/v1",
		Factory: func(cfg ProviderConfig) Provider {
			captured = cfg
			return &stubProvider{name: providerName}
		},
	})
	if err != nil {
		t.Fatalf("RegisterProviderWithDescriptor returned error: %v", err)
	}

	_, err = NewWithProvider(providerName, ProviderConfig{})
	if err != nil {
		t.Fatalf("NewWithProvider returned error: %v", err)
	}

	if captured.BaseURL != "https://example.test/v1" {
		t.Fatalf("base URL = %q, want %q", captured.BaseURL, "https://example.test/v1")
	}
}

func TestRegisterProviderRejectsConflicts(t *testing.T) {
	providerName := strings.ToLower(strings.ReplaceAll(t.Name(), "/", "_"))
	err := RegisterProvider("openai", func(cfg ProviderConfig) Provider {
		return &stubProvider{name: "openai"}
	})
	if err == nil {
		t.Fatal("expected builtin conflict to fail")
	}

	err = RegisterProvider(providerName, func(cfg ProviderConfig) Provider {
		return &stubProvider{name: providerName}
	})
	if err != nil {
		t.Fatalf("first RegisterProvider returned error: %v", err)
	}

	err = RegisterProvider(providerName, func(cfg ProviderConfig) Provider {
		return &stubProvider{name: providerName}
	})
	if err == nil {
		t.Fatal("expected duplicate registration to fail")
	}

	descriptorName := providerName + "_descriptor"
	err = RegisterProviderWithDescriptor(ProviderDescriptor{
		Name: descriptorName,
		Factory: func(cfg ProviderConfig) Provider {
			return &stubProvider{name: descriptorName}
		},
	})
	if err != nil {
		t.Fatalf("first RegisterProviderWithDescriptor returned error: %v", err)
	}

	err = RegisterProviderWithDescriptor(ProviderDescriptor{
		Name: descriptorName,
		Factory: func(cfg ProviderConfig) Provider {
			return &stubProvider{name: descriptorName}
		},
	})
	if err == nil {
		t.Fatal("expected duplicate descriptor registration to fail")
	}
}
