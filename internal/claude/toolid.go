package claude

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

const maxToolUseID = 64

// ToolUseID returns id in a form both Claude protocols accept: Anthropic
// allows [a-zA-Z0-9_-]+ and Bedrock Converse at most 64 characters. A
// conforming id is returned unchanged; any other is rewritten
// deterministically, with a hash of the original appended so distinct ids
// stay distinct. Apply it to tool calls and tool results alike.
func ToolUseID(id string) string {
	if len(id) <= maxToolUseID && strings.IndexFunc(id, invalidIDRune) < 0 {
		return id
	}
	sum := sha256.Sum256([]byte(id))
	suffix := "_" + hex.EncodeToString(sum[:4])
	safe := strings.Map(func(r rune) rune {
		if invalidIDRune(r) {
			return '_'
		}
		return r
	}, id)
	return safe[:min(len(safe), maxToolUseID-len(suffix))] + suffix
}

func invalidIDRune(r rune) bool {
	return !('a' <= r && r <= 'z' || 'A' <= r && r <= 'Z' || '0' <= r && r <= '9' || r == '_' || r == '-')
}
