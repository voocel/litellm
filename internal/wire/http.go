package wire

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"

	"github.com/voocel/litellm"
)

// MaxErrorBody bounds how much of a failed response is read into the error.
const MaxErrorBody = 1 << 20

// Do sends req. A transport failure becomes a network error and a non-2xx
// response an HTTP error with its body closed, as does a 2xx HTML page. On
// success the caller owns resp.Body.
func Do(client litellm.HTTPClient, req *http.Request, provider, operation string) (*http.Response, error) {
	resp, err := client.Do(req)
	if err != nil {
		return nil, litellm.NewNetworkError(provider, operation+" failed", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, MaxErrorBody))
		return nil, HTTPError(provider, resp.StatusCode, resp.Header, string(data))
	}
	if isHTMLPage(resp) {
		resp.Body.Close()
		endpoint := req.URL.Scheme + "://" + req.URL.Host + req.URL.Path
		return nil, litellm.NewError(provider, litellm.ErrorTypeProvider, fmt.Sprintf("%s returned an HTML page instead of an API response; check BaseURL", endpoint), nil)
	}
	return resp, nil
}

// isHTMLPage reports a web page, typically a gateway console served because
// BaseURL lacks the API path, which would otherwise surface as a decode error
// or an empty stream. The body must start with markup, since some servers
// label JSON as text/html; a body that does not is kept for the caller.
func isHTMLPage(resp *http.Response) bool {
	mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if mediaType != "text/html" {
		return false
	}
	br := bufio.NewReader(resp.Body)
	resp.Body = struct {
		io.Reader
		io.Closer
	}{br, resp.Body}
	for n := 1; ; n++ {
		b, err := br.Peek(n)
		if err != nil {
			return false
		}
		switch c := b[n-1]; c {
		case ' ', '\t', '\r', '\n':
		default:
			return c == '<'
		}
	}
}

// ErrorField converts the error member of an OpenAI-style response body or
// stream chunk, which gateways such as OpenRouter send with HTTP 200, or
// returns nil when it is absent or null. A numeric code is the upstream HTTP
// status and is classified as one; otherwise the code, metadata.error_type or
// type is used.
func ErrorField(provider string, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return StreamError(provider, "", text)
	}
	var e struct {
		Code     json.RawMessage `json:"code"`
		Type     string          `json:"type"`
		Message  string          `json:"message"`
		Metadata struct {
			ErrorType string `json:"error_type"`
		} `json:"metadata"`
	}
	_ = json.Unmarshal(raw, &e)
	var status int
	if json.Unmarshal(e.Code, &status) == nil && status >= 400 {
		return HTTPError(provider, status, nil, `{"error":`+string(raw)+`}`)
	}
	var code string
	_ = json.Unmarshal(e.Code, &code)
	message := cmp.Or(e.Message, string(raw))
	return StreamError(provider, cmp.Or(code, e.Metadata.ErrorType, e.Type), message)
}

// HTTPClient returns c, or http.DefaultClient when c is nil.
func HTTPClient(c litellm.HTTPClient) litellm.HTTPClient {
	if c == nil {
		return http.DefaultClient
	}
	return c
}

// DefaultUserAgent identifies the SDK when a config sets no UserAgent.
const DefaultUserAgent = "litellm-go/0.1"

// APIKey resolves the key through fn when set. An empty key is an error only
// when required.
func APIKey(ctx context.Context, key string, fn func(context.Context) (string, error), required bool) (string, error) {
	if fn != nil {
		resolved, err := fn(ctx)
		if err != nil {
			return "", fmt.Errorf("resolve api key: %w", err)
		}
		key = resolved
	}
	if required && key == "" {
		return "", errors.New("api key is required")
	}
	return key, nil
}

// SetHeaders applies user-configured headers, skipping blank values.
func SetHeaders(h http.Header, headers map[string]string) error {
	for name, value := range headers {
		name, value = strings.TrimSpace(name), strings.TrimSpace(value)
		if name == "" {
			return errors.New("header name cannot be empty")
		}
		if value != "" {
			h.Set(name, value)
		}
	}
	return nil
}
