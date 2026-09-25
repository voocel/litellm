package bedrock

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// errInvalidFrame marks a corrupt AWS event stream frame, as opposed to a
// failed read.
var errInvalidFrame = errors.New("invalid event stream frame")

// eventStreamMessage is one application/vnd.amazon.eventstream frame. Only
// string headers are kept; they carry the message, event and exception types.
type eventStreamMessage struct {
	headers map[string]string
	payload []byte
}

// Frame layout: total length, headers length and prelude CRC (uint32 each),
// headers, payload, then a CRC of everything before it. Both CRCs are IEEE.
func readEventStreamMessage(reader *bufio.Reader) (eventStreamMessage, error) {
	prelude := make([]byte, 12)
	if _, err := io.ReadFull(reader, prelude); err != nil {
		return eventStreamMessage{}, err
	}
	totalLength := binary.BigEndian.Uint32(prelude[0:4])
	headersLength := binary.BigEndian.Uint32(prelude[4:8])
	if crc32.ChecksumIEEE(prelude[:8]) != binary.BigEndian.Uint32(prelude[8:12]) {
		return eventStreamMessage{}, fmt.Errorf("%w: prelude checksum mismatch", errInvalidFrame)
	}
	if totalLength < 16 || totalLength > 16*1024*1024 {
		return eventStreamMessage{}, fmt.Errorf("%w: message length %d", errInvalidFrame, totalLength)
	}
	if headersLength > totalLength-16 {
		return eventStreamMessage{}, fmt.Errorf("%w: headers length %d exceeds %d", errInvalidFrame, headersLength, totalLength-16)
	}
	rest := make([]byte, totalLength-12)
	if _, err := io.ReadFull(reader, rest); err != nil {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return eventStreamMessage{}, err
	}
	body := rest[:len(rest)-4]
	checksum := crc32.Update(crc32.ChecksumIEEE(prelude), crc32.IEEETable, body)
	if checksum != binary.BigEndian.Uint32(rest[len(rest)-4:]) {
		return eventStreamMessage{}, fmt.Errorf("%w: message checksum mismatch", errInvalidFrame)
	}
	headers, err := parseEventStreamHeaders(body[:headersLength])
	if err != nil {
		return eventStreamMessage{}, err
	}
	return eventStreamMessage{headers: headers, payload: body[headersLength:]}, nil
}

// Header value sizes by type; byte arrays and strings are length-prefixed.
var headerValueSizes = map[byte]int{0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}

const (
	headerTypeBytes  = 6
	headerTypeString = 7
)

func parseEventStreamHeaders(b []byte) (map[string]string, error) {
	headers := make(map[string]string)
	for len(b) > 0 {
		nameLength := int(b[0])
		if len(b) < 2+nameLength {
			return nil, fmt.Errorf("%w: truncated header name", errInvalidFrame)
		}
		name := string(b[1 : 1+nameLength])
		valueType := b[1+nameLength]
		b = b[2+nameLength:]
		switch valueType {
		case headerTypeBytes, headerTypeString:
			if len(b) < 2 || len(b) < 2+int(binary.BigEndian.Uint16(b)) {
				return nil, fmt.Errorf("%w: truncated header %q", errInvalidFrame, name)
			}
			size := int(binary.BigEndian.Uint16(b))
			if valueType == headerTypeString {
				headers[name] = string(b[2 : 2+size])
			}
			b = b[2+size:]
		default:
			size, ok := headerValueSizes[valueType]
			if !ok || len(b) < size {
				return nil, fmt.Errorf("%w: header %q has invalid type %d", errInvalidFrame, name, valueType)
			}
			b = b[size:]
		}
	}
	return headers, nil
}
