package bedrock

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/retry"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(req *http.Request) (*http.Response, error) { return f(req) }

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestNewRequiresCredentials(t *testing.T) {
	_, err := New(Config{})
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || err.Error() != "bedrock: credentials provider is required" {
		t.Fatalf("err = %v", err)
	}
}

func TestCapabilities(t *testing.T) {
	want := litellm.Capabilities{ThinkingEffort: true, DisableThinking: true, ProviderOptions: []string{
		"additionalModelRequestFields", "additionalModelResponseFieldPaths", "guardrailConfig",
		"performanceConfig", "promptVariables", "requestMetadata",
	}}
	if got := newProvider(t, nil).Capabilities(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Capabilities = %+v", got)
	}
}

func TestChatSignsRequestAndConvertsResponse(t *testing.T) {
	var req *http.Request
	p := newProvider(t, doerFunc(func(r *http.Request) (*http.Response, error) {
		req = r
		return jsonResponse(http.StatusOK, `{
			"output":{"message":{"role":"assistant","content":[{"text":"hello"}]}},
			"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}}`), nil
	}))
	resp, err := p.Chat(context.Background(), &litellm.Request{Model: "anthropic.claude-v1:0", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if req.URL.RawPath != "/model/anthropic.claude-v1%3A0/converse" {
		t.Fatalf("raw path = %q", req.URL.RawPath)
	}
	if auth := req.Header.Get("Authorization"); !strings.Contains(auth, "Credential=AKID/") || !strings.Contains(auth, "/us-west-2/bedrock/aws4_request") {
		t.Fatalf("authorization = %q", auth)
	}
	if req.Header.Get("X-Amz-Security-Token") != "SESSION" {
		t.Fatalf("session token = %q", req.Header.Get("X-Amz-Security-Token"))
	}
	if resp.Text() != "hello" || resp.Model != "anthropic.claude-v1:0" || resp.FinishReason != litellm.FinishReasonStop {
		t.Fatalf("response = %+v", resp)
	}
}

// A retrying HTTPClient resends the signed request, which stays valid.
func TestChatRetriesThroughHTTPClient(t *testing.T) {
	var auths []string
	client := retry.NewHTTPClient(&http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		auths = append(auths, r.Header.Get("Authorization"))
		if body, _ := io.ReadAll(r.Body); !json.Valid(body) {
			t.Fatalf("attempt %d body = %q", len(auths), body)
		}
		if len(auths) == 1 {
			return jsonResponse(http.StatusTooManyRequests, `{"message":"slow down"}`), nil
		}
		return jsonResponse(http.StatusOK, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn"}`), nil
	})}, &retry.Policy{MaxAttempts: 2, InitialDelay: time.Nanosecond})
	resp, err := newProvider(t, client).Chat(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	if len(auths) != 2 || auths[0] == "" || auths[0] != auths[1] || resp.Text() != "ok" {
		t.Fatalf("auths = %q, text = %q", auths, resp.Text())
	}
}

func TestChatReturnsHTTPErrors(t *testing.T) {
	p := newProvider(t, doerFunc(func(*http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusBadRequest, `{"message":"bad model"}`), nil
	}))
	_, err := p.Chat(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
	if litellm.ErrorTypeOf(err) != litellm.ErrorTypeValidation || !strings.Contains(err.Error(), "bad model") {
		t.Fatalf("err = %v", err)
	}
}

func newProvider(t *testing.T, client litellm.HTTPClient) *Provider {
	t.Helper()
	p, err := New(Config{Region: "us-west-2", Credentials: StaticCredentials("AKID", "SECRET", "SESSION"), HTTPClient: client})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
}

type eventFrame struct {
	headers [][2]string
	payload string
}

func event(name, payload string) eventFrame {
	return eventFrame{headers: [][2]string{{":message-type", "event"}, {":event-type", name}}, payload: payload}
}

func exception(name, payload string) eventFrame {
	return eventFrame{headers: [][2]string{{":message-type", "exception"}, {":exception-type", name}}, payload: payload}
}

// eventStream encodes frames with string headers and valid checksums.
func eventStream(frames ...eventFrame) io.ReadCloser {
	var out bytes.Buffer
	for _, frame := range frames {
		var headers bytes.Buffer
		for _, h := range frame.headers {
			headers.WriteByte(byte(len(h[0])))
			headers.WriteString(h[0])
			headers.WriteByte(headerTypeString)
			_ = binary.Write(&headers, binary.BigEndian, uint16(len(h[1])))
			headers.WriteString(h[1])
		}
		msg := prelude(uint32(16+headers.Len()+len(frame.payload)), uint32(headers.Len()))
		msg = append(msg, headers.Bytes()...)
		msg = append(msg, frame.payload...)
		msg = binary.BigEndian.AppendUint32(msg, crc32.ChecksumIEEE(msg))
		out.Write(msg)
	}
	return io.NopCloser(&out)
}

func TestCredentialsResolvedPerCall(t *testing.T) {
	credentials := &countingCredentials{credentials: Credentials{AccessKeyID: "AKID", SecretAccessKey: "SECRET"}}
	var signed int
	client := doerFunc(func(req *http.Request) (*http.Response, error) {
		if strings.Contains(req.Header.Get("Authorization"), "AWS4-HMAC-SHA256 Credential=AKID/") {
			signed++
		}
		return jsonResponse(http.StatusOK, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn"}`), nil
	})
	p, err := New(Config{Credentials: credentials, HTTPClient: client})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := p.Chat(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}}); err != nil {
			t.Fatal(err)
		}
	}
	if signed != 2 || credentials.calls != 2 {
		t.Fatalf("signed = %d, credential calls = %d, want 2 each", signed, credentials.calls)
	}
}

// Credentials that cannot sign fail the call as an auth error, which no
// retry clears, without sending it.
func TestCredentialsThatCannotSign(t *testing.T) {
	for _, tc := range []struct {
		name        string
		credentials CredentialsProvider
	}{
		{"resolving fails", &countingCredentials{err: errors.New("no profile")}},
		{"no secret", StaticCredentials("AKID", "", "")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := doerFunc(func(*http.Request) (*http.Response, error) {
				t.Fatal("request sent")
				return nil, nil
			})
			p, err := New(Config{Credentials: tc.credentials, HTTPClient: client})
			if err != nil {
				t.Fatal(err)
			}
			_, err = p.Chat(context.Background(), &litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}})
			if litellm.ErrorTypeOf(err) != litellm.ErrorTypeAuth || litellm.IsTemporaryError(err) {
				t.Fatalf("err = %v (%s), want a lasting auth error", err, litellm.ErrorTypeOf(err))
			}
		})
	}
}

type countingCredentials struct {
	credentials Credentials
	err         error
	calls       int
}

func (c *countingCredentials) Credentials(context.Context) (Credentials, error) {
	c.calls++
	return c.credentials, c.err
}
