package litellm

import (
	"reflect"
	"testing"
)

type capabilityProvider struct {
	testProvider
	caps Capabilities
}

func (p *capabilityProvider) Capabilities() Capabilities { return p.caps }

func TestClientCapabilities(t *testing.T) {
	provider := &capabilityProvider{
		testProvider: testProvider{name: "test"},
		caps:         Capabilities{Thinking: true, ProviderOptions: []string{"a", "b"}},
	}
	client, err := New(provider)
	if err != nil {
		t.Fatal(err)
	}
	caps, ok := client.Capabilities()
	if !ok || !reflect.DeepEqual(caps, provider.caps) {
		t.Fatalf("caps = %+v", caps)
	}
	caps.ProviderOptions[0] = "mutated"
	if provider.caps.ProviderOptions[0] != "a" {
		t.Fatal("Capabilities shares the provider's option list")
	}

	plain, err := New(&testProvider{name: "plain"})
	if err != nil {
		t.Fatal(err)
	}
	var nilClient *Client
	for name, client := range map[string]*Client{"plain provider": plain, "nil client": nilClient} {
		if caps, ok := client.Capabilities(); ok || !reflect.DeepEqual(caps, Capabilities{}) {
			t.Errorf("%s: caps = %+v, %v; want unknown", name, caps, ok)
		}
	}
}
