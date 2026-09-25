package litellm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestErrorRendering(t *testing.T) {
	for _, tc := range []struct {
		err  *Error
		want string
	}{
		{&Error{Provider: "p", Code: "c", Message: " boom "}, "p: c: boom"},
		{&Error{Message: "boom"}, "boom"},
		{&Error{Provider: "p", Code: "c"}, "p: c"},
		{&Error{Provider: "p", StatusCode: 503, Type: ErrorTypeProvider}, "p: HTTP 503 (provider)"},
		{&Error{StatusCode: 503}, "HTTP 503"},
		{&Error{Type: ErrorTypeValidation}, "validation"},
		{&Error{}, "litellm error"},
		{NewNetworkError("p", "read failed", io.EOF), "p: read failed: EOF"},
		{NewNetworkError("p", "read: EOF", io.EOF), "p: read: EOF"},
		{NewError("p", ErrorTypeProvider, "decode", io.EOF), "p: decode"},
	} {
		if got := tc.err.Error(); got != tc.want {
			t.Errorf("Error() = %q, want %q", got, tc.want)
		}
	}
	if got := WrapError("p", ErrorTypeProvider, errors.New("boom")).Error(); got != "p: boom" {
		t.Errorf("wrapped plain error = %q", got)
	}
}

func TestTemporaryErrorClassification(t *testing.T) {
	for _, tc := range []struct {
		name      string
		err       error
		temporary bool
	}{
		{"unknown_provider", NewError("test", ErrorTypeProvider, "unknown", nil), false},
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
	original := &Error{Type: ErrorTypeProvider, StatusCode: 503, Temporary: true, Message: "unavailable"}
	wrapped := WrapError("test", ErrorTypeProvider, original)
	var typed *Error
	if !errors.As(wrapped, &typed) || typed.Provider != "test" || !typed.Temporary || typed.StatusCode != 503 {
		t.Fatalf("wrapped = %#v", wrapped)
	}
	if original.Provider != "" {
		t.Fatal("WrapError mutated original")
	}
	canceled := WrapError("test", ErrorTypeProvider, fmt.Errorf("transport: %w", context.Canceled))
	if !IsNetworkError(canceled) || IsTemporaryError(canceled) || !errors.Is(canceled, context.Canceled) {
		t.Fatalf("canceled error = %v", canceled)
	}
	deadline := WrapError("test", ErrorTypeProvider, fmt.Errorf("transport: %w", context.DeadlineExceeded))
	if !IsTimeoutError(deadline) || IsTemporaryError(deadline) || !errors.Is(deadline, context.DeadlineExceeded) {
		t.Fatalf("deadline error = %v", deadline)
	}
	// A provider error caused by cancellation is reclassified without
	// repeating its provider prefix.
	read := WrapError("test", ErrorTypeProvider, NewError("test", ErrorTypeProvider, "read stream", context.DeadlineExceeded))
	if !IsTimeoutError(read) || read.Error() != "test: read stream: context deadline exceeded" {
		t.Fatalf("read error = %v", read)
	}
}
