package bedrock

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"testing"

	"github.com/voocel/litellm/internal/testgolden"
)

// testdata/bedrock/*.bin are encoded by aws-sdk-go-v2 aws/protocol/eventstream
// (v1.7.20) with the headers and bare payloads Bedrock ConverseStream sends.
func TestReadEventStreamMessageReadsHeadersAndPayload(t *testing.T) {
	reader := bufio.NewReader(bytes.NewReader(testgolden.ReadFixture(t, "../../testdata/bedrock/eventstream.bin")))
	if _, err := readEventStreamMessage(reader); err != nil {
		t.Fatalf("read messageStart: %v", err)
	}
	msg, err := readEventStreamMessage(reader)
	if err != nil {
		t.Fatalf("readEventStreamMessage: %v", err)
	}
	if msg.headers[":message-type"] != "event" || msg.headers[":event-type"] != "contentBlockDelta" {
		t.Fatalf("headers = %v", msg.headers)
	}
	if string(msg.payload) != `{"contentBlockIndex":0,"delta":{"text":"hel"},"p":"abcdefgh"}` {
		t.Fatalf("payload = %s", msg.payload)
	}
}

func TestReadEventStreamMessageRejectsCorruptFrames(t *testing.T) {
	valid := testgolden.ReadFixture(t, "../../testdata/bedrock/eventstream.bin")
	flipped := func(offset int) []byte {
		frame := append([]byte(nil), valid...)
		frame[offset] ^= 0xff
		return frame
	}
	tests := map[string][]byte{
		"prelude checksum": flipped(0),
		"message checksum": flipped(12),
		"message length":   prelude(15, 0),
		"headers length":   append(prelude(16, 1), 0, 0, 0, 0),
	}
	for name, frame := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := readEventStreamMessage(bufio.NewReader(bytes.NewReader(frame)))
			if !errors.Is(err, errInvalidFrame) {
				t.Fatalf("err = %v, want errInvalidFrame", err)
			}
		})
	}
}

func prelude(totalLength, headersLength uint32) []byte {
	b := make([]byte, 12)
	binary.BigEndian.PutUint32(b[0:4], totalLength)
	binary.BigEndian.PutUint32(b[4:8], headersLength)
	binary.BigEndian.PutUint32(b[8:12], crc32.ChecksumIEEE(b[:8]))
	return b
}
