package idle

import (
	"io"
	"strings"
	"testing"
	"time"
)

func TestReadWaitingTooLongFails(t *testing.T) {
	r, w := io.Pipe()
	defer w.Close()
	body := Watch(r, 20*time.Millisecond)
	go func() {
		w.Write([]byte("a"))
		time.Sleep(10 * time.Millisecond)
		w.Write([]byte("b")) // within the timeout of the read waiting for it
	}()
	buf := make([]byte, 1)
	for _, want := range "ab" {
		if n, err := body.Read(buf); err != nil || n != 1 || rune(buf[0]) != want {
			t.Fatalf("read %q, %v", buf[:n], err)
		}
	}
	if _, err := body.Read(buf); err == nil || !strings.Contains(err.Error(), "no data for 20ms") {
		t.Fatalf("idle read: %v", err)
	}
	if _, err := body.Read(buf); err == nil {
		t.Fatal("a read after the timeout succeeded")
	}
}

// Only a read waiting counts: data a consumer is slow to read is no sign of
// a connection that hung.
func TestSlowConsumerIsNotIdle(t *testing.T) {
	body := Watch(io.NopCloser(strings.NewReader("data")), 10*time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	if data, err := io.ReadAll(body); err != nil || string(data) != "data" {
		t.Fatalf("read %q, %v", data, err)
	}
}
