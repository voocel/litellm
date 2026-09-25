package wire

import (
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ParseRetryAfter returns the delay described by value, either delay-seconds or an
// HTTP-date relative to now. Absent, invalid or past values yield zero.
func ParseRetryAfter(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		// Beyond the Duration range the value is unusable, so invalid.
		if seconds > 0 && seconds <= int64(math.MaxInt64/time.Second) {
			return time.Duration(seconds) * time.Second
		}
		return 0
	}
	if when, err := http.ParseTime(value); err == nil {
		if delay := when.Sub(now); delay > 0 {
			return delay
		}
	}
	return 0
}
