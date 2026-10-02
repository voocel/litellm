// Package bedrock connects to the Amazon Bedrock Converse API.
package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/internal/wire"
)

// Config configures Converse. HTTPClient, when set, sends the SigV4-signed
// requests; wrap it with retry.NewHTTPClient to retry.
type Config struct {
	// Name identifies the provider in responses, errors and ProviderState,
	// "bedrock" by default.
	Name string
	// Region selects the endpoint and signing region, us-east-1 by default.
	Region string
	// BaseURL overrides the bedrock-runtime endpoint.
	BaseURL string
	// Credentials is required and resolved for every request.
	Credentials CredentialsProvider
	HTTPClient  litellm.HTTPClient
}

// Credentials are AWS credentials; SessionToken is for temporary ones.
type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// CredentialsProvider supplies credentials, allowing rotation.
type CredentialsProvider interface {
	Credentials(context.Context) (Credentials, error)
}

type staticCredentials struct {
	credentials Credentials
}

// StaticCredentials returns a CredentialsProvider with fixed credentials.
func StaticCredentials(accessKeyID, secretAccessKey, sessionToken string) CredentialsProvider {
	return staticCredentials{credentials: Credentials{
		AccessKeyID:     accessKeyID,
		SecretAccessKey: secretAccessKey,
		SessionToken:    sessionToken,
	}}
}

func (p staticCredentials) Credentials(context.Context) (Credentials, error) {
	return p.credentials, nil
}

const defaultRegion = "us-east-1"

// Provider implements litellm.Provider and litellm.CapabilityProvider.
type Provider struct {
	cfg Config
}

// New returns a Provider for cfg.
func New(cfg Config) (*Provider, error) {
	if cfg.Name == "" {
		cfg.Name = "bedrock"
	}
	if cfg.Region == "" {
		cfg.Region = defaultRegion
	}
	if cfg.Credentials == nil {
		return nil, litellm.NewError(cfg.Name, litellm.ErrorTypeValidation, "credentials provider is required", nil)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", cfg.Region)
	}
	cfg.HTTPClient = &http.Client{Transport: newSigningTransport(cfg.Credentials, cfg.Region, clientTransport{client: wire.HTTPClient(cfg.HTTPClient)})}
	return &Provider{cfg: cfg}, nil
}

// Name returns the configured name.
func (p *Provider) Name() string {
	return p.cfg.Name
}

// Capabilities reports the static protocol facts.
func (p *Provider) Capabilities() litellm.Capabilities {
	return litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: sortedOptions()}
}

// Chat sends a Converse request; Request.Model is the model or inference
// profile ID.
func (p *Provider) Chat(ctx context.Context, req *litellm.Request) (*litellm.Response, error) {
	resp, err := p.post(ctx, req, "converse")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, litellm.NewNetworkError(p.Name(), "read response failed", err)
	}
	var parsed response
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode response", err)
	}
	out := convertResponse(&parsed, p.Name(), req.Model)
	out.Raw = data
	return out, nil
}

// Stream sends a ConverseStream request.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	resp, err := p.post(ctx, req, "converse-stream")
	if err != nil {
		return nil, err
	}
	return newStream(resp, p.Name(), req.Model), nil
}

func (p *Provider) post(ctx context.Context, req *litellm.Request, operation string) (*http.Response, error) {
	body, err := buildRequest(req, p.Name())
	if err != nil {
		return nil, litellm.WrapError(p.Name(), litellm.ErrorTypeValidation, err)
	}
	endpoint, rawPath, err := runtimeEndpoint(p.cfg.BaseURL, req.Model, operation)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeValidation, "invalid base url", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create request", err)
	}
	httpReq.URL.RawPath = rawPath
	return wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "request")
}

type clientTransport struct {
	client litellm.HTTPClient
}

func (t clientTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.client.Do(req)
}

func runtimeEndpoint(baseURL, model, operation string) (string, string, error) {
	endpoint, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return "", "", err
	}
	pathPrefix := strings.TrimRight(endpoint.Path, "/")
	rawPrefix := strings.TrimRight(endpoint.EscapedPath(), "/")
	path := pathPrefix + "/model/" + model + "/" + operation
	rawPath := rawPrefix + "/model/" + awsEscapePath(model, true) + "/" + operation
	endpoint.Path = path
	endpoint.RawPath = rawPath
	return endpoint.String(), rawPath, nil
}
