package openai

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

// Gateways that repeat id and name on every chunk, send the name late, or
// send parallel calls whole at one index still yield one call per call.
func TestStreamToolCallsByGatewayShape(t *testing.T) {
	cases := []struct {
		name, sse string
		want      []string
	}{
		{"repeated id and name", `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":""}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"a\":"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`, []string{`c1 f {"a":1}`}},
		{"late name", `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"arguments":"{\"a\":"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"1}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`, []string{`c1 f {"a":1}`}},
		{"parallel at index 0", `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c2","function":{"name":"g","arguments":"{\"b\":2}"}}]},"finish_reason":"tool_calls"}]}
data: [DONE]
`, []string{`c1 f {"a":1}`, `c2 g {"b":2}`}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := &http.Response{Body: io.NopCloser(strings.NewReader(c.sse))}
			s := newStream(resp, &litellm.Request{Model: "m"})
			var order []string
			names, args := map[string]string{}, map[string]string{}
		read:
			for {
				ev, err := s.Next()
				if err != nil {
					t.Fatal(err)
				}
				switch e := ev.(type) {
				case litellm.ToolUseStart:
					order = append(order, e.ID)
					names[e.ID] = e.Name
				case litellm.ToolUseDelta:
					args[e.ID] += string(e.ArgumentsDelta)
				case litellm.DoneEvent:
					break read
				}
			}
			var got []string
			for _, id := range order {
				got = append(got, id+" "+names[id]+" "+args[id])
			}
			if strings.Join(got, "|") != strings.Join(c.want, "|") {
				t.Fatalf("got %q want %q", got, c.want)
			}
		})
	}
}
