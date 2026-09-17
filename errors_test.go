package litellm

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestTemporaryErrorClassification(t *testing.T) {
	for _, tc := range []struct {
		name      string
		err       error
		temporary bool
	}{
		{"rate_limit", NewHTTPError("test", 429, "busy"), true},
		{"unavailable", NewHTTPError("test", 503, "busy"), true},
		{"timeout", NewHTTPError("test", 408, "timeout"), true},
		{"overloaded", NewHTTPError("test", 529, "busy"), true},
		{"unsupported", NewHTTPError("test", 501, "unsupported"), false},
		{"bad_request", NewHTTPError("test", 400, "bad input"), false},
		{"moderation", NewHTTPError("test", 503, `{"error":{"code":"content_filter","message":"blocked"}}`), false},
		{"unknown_provider", NewProviderError("test", ErrorTypeProvider, "unknown"), false},
		{"network_outcome_unknown", NewNetworkError("test", "read failed", errors.New("EOF")), true},
		{"caller_canceled", NewNetworkError("test", "canceled", context.Canceled), false},
		{"caller_deadline", NewNetworkError("test", "deadline", context.DeadlineExceeded), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsTemporaryError(fmt.Errorf("call: %w", tc.err)); got != tc.temporary {
				t.Fatalf("IsTemporaryError = %v, want %v", got, tc.temporary)
			}
		})
	}
}

func TestWrapErrorPreservesCauseAndClassification(t *testing.T) {
	original := NewHTTPError("", 503, "unavailable")
	wrapped := WrapError(original, "test")
	var typed *LiteLLMError
	if !errors.As(wrapped, &typed) || typed.Provider != "test" || !typed.Temporary || typed.StatusCode != 503 {
		t.Fatalf("wrapped = %#v", wrapped)
	}
	if original.Provider != "" {
		t.Fatal("WrapError mutated original")
	}
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		if err := WrapError(fmt.Errorf("transport: %w", cause), "test"); !errors.Is(err, cause) {
			t.Fatalf("lost cause %v: %v", cause, err)
		}
	}
}
