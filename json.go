package litellm

import "encoding/json"

// ParsePartialJSON attempts to parse potentially incomplete JSON from streaming.
// It tries standard parsing first, then attempts to complete truncated JSON
// by closing open strings, arrays, and objects.
// Returns nil if the input is empty; returns an empty map if parsing fails entirely.
func ParsePartialJSON(data string) any {
	if len(data) == 0 {
		return nil
	}

	// Fast path: try standard parse (works for complete JSON)
	var result any
	if err := json.Unmarshal([]byte(data), &result); err == nil {
		return result
	}

	// Slow path: try to complete truncated JSON
	completed := completeJSON(data)
	if err := json.Unmarshal([]byte(completed), &result); err == nil {
		return result
	}

	return map[string]any{}
}

// completeJSON attempts to close truncated JSON by tracking parse state
// and appending the necessary closing tokens.
func completeJSON(s string) string {
	var (
		inString bool
		escaped  bool
		stack    []byte // tracks open '{' and '['
	)

	// Find the last valid position — trim trailing whitespace first
	end := len(s)
	for end > 0 && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	if end == 0 {
		return "{}"
	}
	s = s[:end]

	for i := 0; i < len(s); i++ {
		c := s[i]

		if escaped {
			escaped = false
			continue
		}

		if inString {
			switch c {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}

		switch c {
		case '"':
			inString = true
		case '{':
			stack = append(stack, '}')
		case '[':
			stack = append(stack, ']')
		case '}', ']':
			if len(stack) > 0 && stack[len(stack)-1] == c {
				stack = stack[:len(stack)-1]
			}
		}
	}

	buf := make([]byte, 0, len(s)+len(stack)+2)
	buf = append(buf, s...)

	// Close unclosed string
	if inString || escaped {
		buf = append(buf, '"')
	}

	// Iteratively trim trailing fragments that don't form valid JSON values:
	// colons, commas, and bare object keys (a quoted string preceded by , or {).
	for {
		n := len(buf)
		if n == 0 {
			break
		}
		last := buf[n-1]

		// Remove trailing comma or colon
		if last == ',' || last == ':' {
			buf = buf[:n-1]
			continue
		}

		// Remove bare object key: "..." preceded by , or {
		if last == '"' {
			// Find the matching opening quote (skip escaped quotes)
			i := n - 2
			for i >= 0 {
				if buf[i] == '"' && (i == 0 || buf[i-1] != '\\') {
					break
				}
				i--
			}
			if i >= 0 {
				// Check what's before the opening quote (skip whitespace)
				before := i - 1
				for before >= 0 && (buf[before] == ' ' || buf[before] == '\t' || buf[before] == '\n' || buf[before] == '\r') {
					before--
				}
				if before >= 0 && buf[before] == ',' {
					buf = buf[:before]
					continue
				}
				if before >= 0 && buf[before] == '{' {
					buf = buf[:before+1]
					continue
				}
			}
		}

		break
	}

	// Close all open containers in reverse order
	for i := len(stack) - 1; i >= 0; i-- {
		buf = append(buf, stack[i])
	}

	return string(buf)
}
