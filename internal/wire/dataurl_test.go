package wire

import "testing"

func TestParseDataURL(t *testing.T) {
	tests := []struct {
		url, mime, data string
		ok              bool
	}{
		{"data:image/png;base64,AAAA", "image/png", "AAAA", true},
		{"data:image/png,AAAA", "", "", false},
		{"data:text/plain;charset=utf-8,hi", "", "", false},
		{"https://example.com/a.png", "", "", false},
	}
	for _, tt := range tests {
		mime, data, ok := ParseDataURL(tt.url)
		if mime != tt.mime || data != tt.data || ok != tt.ok {
			t.Errorf("ParseDataURL(%q) = %q, %q, %v", tt.url, mime, data, ok)
		}
	}
}
