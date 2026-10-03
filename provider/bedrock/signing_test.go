package bedrock

import (
	"bytes"
	"net/http"
	"strings"
	"testing"
)

func TestSignRequestSetsSigV4Headers(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude/converse", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	signRequest(req, []byte(`{}`), Credentials{
		AccessKeyID:     "AKID",
		SecretAccessKey: "SECRET",
		SessionToken:    "SESSION",
	}, "us-west-2")
	if req.Header.Get("X-Amz-Security-Token") != "SESSION" {
		t.Fatalf("session token = %q", req.Header.Get("X-Amz-Security-Token"))
	}
	if req.Header.Get("X-Amz-Content-Sha256") != "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" {
		t.Fatalf("payload hash = %q", req.Header.Get("X-Amz-Content-Sha256"))
	}
	auth := req.Header.Get("Authorization")
	for _, want := range []string{
		"AWS4-HMAC-SHA256 Credential=AKID/",
		"/us-west-2/bedrock/aws4_request",
		"SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token",
		"Signature=",
	} {
		if !strings.Contains(auth, want) {
			t.Fatalf("authorization header missing %q: %s", want, auth)
		}
	}
}

func TestAWSCanonicalPathEscapesBedrockModelID(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/converse", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	canonicalURI := awsEscapePath(req.URL.EscapedPath(), false)
	if canonicalURI != "/model/anthropic.claude-3-5-sonnet-20240620-v1%253A0/converse" {
		t.Fatalf("canonicalURI = %q", canonicalURI)
	}
}

func TestRuntimeURLEscapesModelIDAsSinglePathSegment(t *testing.T) {
	endpoint, rawPath, err := runtimeEndpoint("https://bedrock-runtime.us-west-2.amazonaws.com", "arn:aws:bedrock:us-west-2:123456789012:prompt/PROMPT12345:1", "converse")
	if err != nil {
		t.Fatalf("runtimeEndpoint: %v", err)
	}
	want := "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Aprompt%2FPROMPT12345%3A1/converse"
	if endpoint != want {
		t.Fatalf("endpoint = %q, want %q", endpoint, want)
	}
	wantRawPath := "/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Aprompt%2FPROMPT12345%3A1/converse"
	if rawPath != wantRawPath {
		t.Fatalf("rawPath = %q, want %q", rawPath, wantRawPath)
	}
}
