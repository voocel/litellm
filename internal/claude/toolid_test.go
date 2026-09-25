package claude

import (
	"regexp"
	"strings"
	"testing"
)

func TestToolUseID(t *testing.T) {
	valid := regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)
	long := strings.Repeat("a", 65)
	for _, id := range []string{"toolu_01A-b", strings.Repeat("a", 64)} {
		if got := ToolUseID(id); got != id {
			t.Errorf("ToolUseID(%q) = %q, want unchanged", id, got)
		}
	}
	seen := map[string]string{}
	for _, id := range []string{"functions.lookup:0", "functions.lookup:1", "call/é", long, long + "b"} {
		got := ToolUseID(id)
		if !valid.MatchString(got) || got != ToolUseID(id) {
			t.Errorf("ToolUseID(%q) = %q, want a stable conforming id", id, got)
		}
		if prev, ok := seen[got]; ok {
			t.Errorf("%q and %q both map to %q", prev, id, got)
		}
		seen[got] = id
	}
}
