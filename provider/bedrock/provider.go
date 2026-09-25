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
	// Region selects the endpoints and signing region, us-east-1 by default.
	Region string
	// BaseURL overrides the bedrock-runtime endpoint and ControlPlaneBaseURL
	// the bedrock endpoint used by ListModels.
	BaseURL             string
	ControlPlaneBaseURL string
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

// Provider implements litellm.Provider, litellm.CapabilityProvider and
// litellm.ModelLister.
type Provider struct {
	cfg Config
}

// New returns a Provider for cfg.
func New(cfg Config) (*Provider, error) {
	if cfg.Region == "" {
		cfg.Region = defaultRegion
	}
	if cfg.Credentials == nil {
		return nil, litellm.NewError("bedrock", litellm.ErrorTypeValidation, "credentials provider is required", nil)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", cfg.Region)
	}
	if cfg.ControlPlaneBaseURL == "" {
		cfg.ControlPlaneBaseURL = fmt.Sprintf("https://bedrock.%s.amazonaws.com", cfg.Region)
	}
	cfg.HTTPClient = &http.Client{Transport: newSigningTransport(cfg.Credentials, cfg.Region, clientTransport{client: wire.HTTPClient(cfg.HTTPClient)})}
	return &Provider{cfg: cfg}, nil
}

// Name returns "bedrock".
func (p *Provider) Name() string {
	return "bedrock"
}

// Capabilities reports the static protocol facts.
func (p *Provider) Capabilities() litellm.Capabilities {
	return litellm.Capabilities{Thinking: true, DisableThinking: true, ThinkingEffort: true, ThinkingBudget: true, ProviderOptions: sortedOptions()}
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
	out := convertResponse(&parsed, req.Model)
	out.Raw = data
	return out, nil
}

// Stream sends a ConverseStream request.
func (p *Provider) Stream(ctx context.Context, req *litellm.Request) (litellm.Stream, error) {
	resp, err := p.post(ctx, req, "converse-stream")
	if err != nil {
		return nil, err
	}
	return newStream(resp, req.Model), nil
}

func (p *Provider) post(ctx context.Context, req *litellm.Request, operation string) (*http.Response, error) {
	body, err := buildRequest(req)
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

// ListModels lists the foundation models in the region.
func (p *Provider) ListModels(ctx context.Context) ([]litellm.ModelInfo, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.cfg.ControlPlaneBaseURL, "/")+"/foundation-models", nil)
	if err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeInternal, "create models request", err)
	}
	resp, err := wire.Do(p.cfg.HTTPClient, httpReq, p.Name(), "models request")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var payload modelList
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, litellm.NewError(p.Name(), litellm.ErrorTypeProvider, "decode models response", err)
	}
	models := make([]litellm.ModelInfo, 0, len(payload.ModelSummaries))
	for _, item := range payload.ModelSummaries {
		name := item.ModelName
		if name == "" {
			name = item.ModelID
		}
		models = append(models, litellm.ModelInfo{
			ID:               item.ModelID,
			Name:             name,
			Provider:         item.ProviderName,
			InputTokenLimit:  item.InputTokenLimit,
			OutputTokenLimit: item.OutputTokenLimit,
		})
	}
	return models, nil
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
