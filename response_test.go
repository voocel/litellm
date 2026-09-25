package litellm

import "testing"

func TestUsageAccessorsDistinguishUnknownFromZero(t *testing.T) {
	u := Usage{InputTokens: new(0), OutputTokens: new(7)}
	if n, ok := u.Input(); n != 0 || !ok {
		t.Fatalf("Input() = %d, %v; want known zero", n, ok)
	}
	if n, ok := u.Output(); n != 7 || !ok {
		t.Fatalf("Output() = %d, %v", n, ok)
	}
	if _, ok := u.CacheRead(); ok {
		t.Fatal("CacheRead() reported an unknown count as known")
	}
}
