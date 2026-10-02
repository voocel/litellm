package gateway

import (
	"testing"
	"time"
)

// SetHeartbeatInterval sets the heartbeat interval for the rest of t,
// including its cleanups registered later, such as closing test servers.
func SetHeartbeatInterval(t testing.TB, d time.Duration) {
	old := heartbeatInterval
	heartbeatInterval = d
	t.Cleanup(func() { heartbeatInterval = old })
}
