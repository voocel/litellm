package providers

import "testing"

func TestCompatFindReasoningSupportsSummaryText(t *testing.T) {
	compat := &Compat{}

	reasoning, field := compat.findReasoning(map[string]any{
		"reasoning_summary": map[string]any{"text": "summary"},
	})
	if reasoning != "summary" {
		t.Fatalf("reasoning = %q, want %q", reasoning, "summary")
	}
	if field != "reasoning_summary" {
		t.Fatalf("field = %q, want %q", field, "reasoning_summary")
	}
}
