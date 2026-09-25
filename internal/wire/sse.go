package wire

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/voocel/litellm"
)

// maxLine bounds one line; inline media can make single events several MiB.
const maxLine = 64 << 20

// SSEEvent is one data line with the name from a preceding "event:" line.
type SSEEvent struct {
	Name string
	Data string
}

// SSEReader reads the Server-Sent Events framing used by LLM streaming APIs.
// Each "data:" line is one payload: providers send single-line JSON and many
// gateways omit the blank line between events, so lines are not joined.
type SSEReader struct {
	r        *bufio.Reader
	provider string
	name     string
	// AcceptBare returns lines outside SSE framing as Data, for endpoints
	// that may stream bare JSON instead.
	AcceptBare bool
}

// NewSSEReader returns a reader over r; provider is used in its errors.
func NewSSEReader(r io.Reader, provider string) *SSEReader {
	return &SSEReader{r: bufio.NewReaderSize(r, 64<<10), provider: provider}
}

// Next returns the next non-empty payload. It returns io.EOF at the end of the
// body and litellm errors for read failures and oversized lines.
func (r *SSEReader) Next() (SSEEvent, error) {
	for {
		line, err := r.readLine()
		if err != nil {
			return SSEEvent{}, err
		}
		switch {
		case line == "":
			r.name = ""
		case line[0] == ':':
		case strings.HasPrefix(line, "event:"):
			r.name = strings.TrimSpace(line[len("event:"):])
		case strings.HasPrefix(line, "data:"):
			data := strings.TrimSpace(line[len("data:"):])
			if data == "" {
				continue
			}
			name := r.name
			r.name = ""
			return SSEEvent{Name: name, Data: data}, nil
		case r.AcceptBare && !isField(line):
			if data := strings.TrimSpace(line); data != "" {
				return SSEEvent{Data: data}, nil
			}
		}
	}
}

func (r *SSEReader) readLine() (string, error) {
	var buf []byte
	for {
		chunk, err := r.r.ReadSlice('\n')
		if len(buf)+len(chunk) > maxLine {
			return "", litellm.NewError(r.provider, litellm.ErrorTypeProvider, fmt.Sprintf("stream line exceeds %d bytes", maxLine), nil)
		}
		buf = append(buf, chunk...)
		switch {
		case errors.Is(err, bufio.ErrBufferFull):
			continue
		case errors.Is(err, io.EOF):
			if len(buf) == 0 {
				return "", io.EOF
			}
		case err != nil:
			return "", litellm.NewNetworkError(r.provider, "stream read error", err)
		}
		return string(bytes.TrimRight(buf, "\r\n")), nil
	}
}

func isField(line string) bool {
	for _, field := range []string{"id:", "retry:"} {
		if strings.HasPrefix(line, field) {
			return true
		}
	}
	return false
}
