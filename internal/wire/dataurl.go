package wire

import "strings"

// ParseDataURL splits "data:<mime>;base64,<data>" into its MIME type and base64
// payload. Other forms, including non-base64 data URLs, report false.
func ParseDataURL(url string) (mimeType, data string, ok bool) {
	rest, found := strings.CutPrefix(url, "data:")
	if !found {
		return "", "", false
	}
	mimeType, rest, found = strings.Cut(rest, ";")
	if !found {
		return "", "", false
	}
	if data, ok = strings.CutPrefix(rest, "base64,"); ok {
		return mimeType, data, true
	}
	return "", "", false
}
