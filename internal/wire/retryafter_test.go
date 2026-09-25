package wire

import (
	"net/http"
	"testing"
	"time"
)

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		value string
		want  time.Duration
	}{
		{"", 0},
		{"7", 7 * time.Second},
		{" 3 ", 3 * time.Second},
		{"0", 0},
		{"-1", 0},
		{"soon", 0},
		{now.Add(90 * time.Second).Format(http.TimeFormat), 90 * time.Second},
		{now.Add(-time.Minute).Format(http.TimeFormat), 0},
	} {
		if got := ParseRetryAfter(tc.value, now); got != tc.want {
			t.Errorf("ParseRetryAfter(%q) = %v, want %v", tc.value, got, tc.want)
		}
	}
}
