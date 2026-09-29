package providers

import (
	"context"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/bedrock"
)

// Every name builds the provider of that name.
func TestNamesBuild(t *testing.T) {
	names := Names()
	if len(names) != 13 || !slices.IsSorted(names) {
		t.Fatalf("names = %v", names)
	}
	cfg := Config{APIKey: "k", BaseURL: "https://example.test", Credentials: bedrock.StaticCredentials("id", "secret", "")}
	for _, name := range names {
		p, err := New(name, cfg)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if p.Name() != name {
			t.Fatalf("New(%q) built %q", name, p.Name())
		}
	}
}

func TestNewUnknown(t *testing.T) {
	if p, err := New("nope", Config{}); p != nil || !litellm.IsValidationError(err) {
		t.Fatalf("New = %v, %v", p, err)
	}
	if p, err := New("bedrock", Config{}); p != nil || !litellm.IsValidationError(err) {
		t.Fatalf("bedrock without credentials = %v, %v", p, err)
	}
}

type captureClient struct {
	req  *http.Request
	body string
}

func (c *captureClient) Do(req *http.Request) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	c.req, c.body = req, string(body)
	return &http.Response{StatusCode: http.StatusBadRequest, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{}`))}, nil
}

// The shared settings reach the wire, and unlisted options pass only when
// allowed.
func TestNewPassesConfig(t *testing.T) {
	options, err := litellm.NewProviderOptions(map[string]any{"vendor_field": 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, allow := range []bool{false, true} {
		capture := &captureClient{}
		p, err := New("deepseek", Config{
			APIKeyFunc:                  func(context.Context) (string, error) { return "k", nil },
			BaseURL:                     "https://example.test/v1",
			HTTPClient:                  capture,
			UserAgent:                   "app/1",
			Headers:                     map[string]string{"X-Test": "1"},
			AllowUnknownProviderOptions: allow,
		})
		if err != nil {
			t.Fatal(err)
		}
		client, err := litellm.New(p)
		if err != nil {
			t.Fatal(err)
		}
		_, err = client.Chat(context.Background(), litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}, ProviderOptions: options})
		if !allow {
			if capture.req != nil || !litellm.IsValidationError(err) {
				t.Fatalf("unlisted option sent: %v", err)
			}
			continue
		}
		r := capture.req
		if r == nil || !strings.HasPrefix(r.URL.String(), "https://example.test/v1/") || r.Header.Get("Authorization") != "Bearer k" ||
			r.Header.Get("User-Agent") != "app/1" || r.Header.Get("X-Test") != "1" || !strings.Contains(capture.body, `"vendor_field":1`) {
			t.Fatalf("request = %v %v, body %s", r.URL, r.Header, capture.body)
		}
	}
}
