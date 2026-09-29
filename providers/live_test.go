package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/retry"
)

// TestLive calls vendor APIs, for the targets whose key is set, when
// LITELLM_LIVE=1. With LITELLM_RECORD=1 each passing scenario also saves its
// responses under testdata/live/<provider>/<scenario>, which TestReplay runs
// offline, so a recorded vendor stays tested against its real wire format.
func TestLive(t *testing.T) {
	if os.Getenv("LITELLM_LIVE") != "1" {
		t.Skip("set LITELLM_LIVE=1 to call vendor APIs")
	}
	record := os.Getenv("LITELLM_RECORD") == "1"
	for _, tg := range targets {
		t.Run(tg.provider, func(t *testing.T) {
			key := os.Getenv(tg.keyEnv)
			if key == "" {
				t.Skipf("%s is not set", tg.keyEnv)
			}
			for _, sc := range scenarios {
				t.Run(sc.name, func(t *testing.T) {
					rec := &recorder{}
					sc.run(t, newClient(t, tg.provider, Config{APIKey: key, BaseURL: os.Getenv(tg.urlEnv), HTTPClient: rec}), tg)
					if record && !t.Failed() {
						rec.save(t, filepath.Join("testdata", "live", tg.provider, sc.name))
					}
				})
			}
		})
	}
}

// TestReplay runs the scenarios against the recorded responses.
func TestReplay(t *testing.T) {
	for _, tg := range targets {
		for _, sc := range scenarios {
			files, err := filepath.Glob(filepath.Join("testdata", "live", tg.provider, sc.name, "*"))
			if err != nil || len(files) == 0 {
				continue
			}
			t.Run(tg.provider+"/"+sc.name, func(t *testing.T) {
				server := replayServer(t, files)
				sc.run(t, newClient(t, tg.provider, Config{APIKey: "replay", BaseURL: server.URL}), tg)
			})
		}
	}
}

type target struct {
	provider string
	model    string
	keyEnv   string
	urlEnv   string
	// pricedUsage: the vendor reports or implies every token count, cache
	// reads and writes included, so usage can be priced.
	pricedUsage bool
}

var targets = []target{
	{provider: "anthropic", model: "claude-haiku-4-5", keyEnv: "ANTHROPIC_API_KEY", urlEnv: "ANTHROPIC_BASE_URL"},
	{provider: "deepseek", model: "deepseek-flash", keyEnv: "DEEPSEEK_API_KEY", urlEnv: "DEEPSEEK_BASE_URL", pricedUsage: true},
	{provider: "gemini", model: "gemini-3.7-flash", keyEnv: "GEMINI_API_KEY", urlEnv: "GEMINI_BASE_URL"},
	{provider: "glm", model: "glm-5.3-flash", keyEnv: "GLM_API_KEY", urlEnv: "GLM_BASE_URL", pricedUsage: true},
	{provider: "openai", model: "gpt-5-mini", keyEnv: "OPENAI_API_KEY", urlEnv: "OPENAI_BASE_URL"},
	{provider: "qwen", model: "qwen3.8-max-0902", keyEnv: "QWEN_API_KEY", urlEnv: "QWEN_BASE_URL", pricedUsage: true},
}

var scenarios = []struct {
	name string
	run  func(*testing.T, *litellm.Client, target)
}{
	{"chat", testChat},
	{"stream", testStream},
	{"tools", testTools},
	{"cache", testCache},
}

func testChat(t *testing.T, c *litellm.Client, tg target) {
	resp, err := c.Chat(callContext(t), request(tg, userText("Reply with the single word: pong")))
	if err != nil {
		t.Fatal(err)
	}
	checkAnswer(t, resp, tg)
}

func testStream(t *testing.T, c *litellm.Client, tg target) {
	stream, err := c.Stream(callContext(t), request(tg, userText("Reply with the single word: pong")))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	checkAnswer(t, resp, tg)
}

// testTools sends the tool call turn back as produced, reasoning state
// included, as vendors that replay reasoning require.
func testTools(t *testing.T, c *litellm.Client, tg target) {
	req := request(tg, userText("What is the weather in Paris right now? Use the tool."))
	req.Tools = []litellm.Tool{{
		Name:        "get_weather",
		Description: "Get the current weather of a city.",
		Parameters:  litellm.Schema(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}}
	resp, err := c.Chat(callContext(t), req)
	if err != nil {
		t.Fatal(err)
	}
	var call *litellm.ToolUseBlock
	for _, block := range resp.Blocks {
		if b, ok := block.(litellm.ToolUseBlock); ok {
			call = &b
		}
	}
	if resp.FinishReason != litellm.FinishReasonToolCall || call == nil || call.Name != "get_weather" || !json.Valid(call.Arguments) {
		t.Fatalf("finish %q, tool call %+v", resp.FinishReason, call)
	}
	checkUsage(t, resp.Usage, tg)
	req.Messages = append(req.Messages,
		litellm.Message{Role: litellm.RoleAssistant, Blocks: resp.Blocks},
		litellm.Message{Role: litellm.RoleTool, Blocks: []litellm.Block{litellm.ToolResultBlock{
			ToolUseID: call.ID,
			Content:   []litellm.Block{litellm.TextBlock{Text: "Sunny, 22°C"}},
		}}},
	)
	final, err := c.Chat(callContext(t), req)
	if err != nil {
		t.Fatal(err)
	}
	checkAnswer(t, final, tg)
}

// testCache sends a long prefix twice. A hit is logged rather than required:
// vendors cache on a best-effort basis.
func testCache(t *testing.T, c *litellm.Client, tg target) {
	var prefix strings.Builder
	for i := range 200 {
		fmt.Fprintf(&prefix, "Fact %d: the Yangtze, about 6,300 km long, is the longest river in China and flows into the East China Sea.\n", i+1)
	}
	system := litellm.Message{Role: litellm.RoleSystem, Blocks: []litellm.Block{
		litellm.TextBlock{Text: prefix.String(), Cache: &litellm.CacheControl{}},
	}}
	req := request(tg, system, userText("In one word, which river is the longest in China?"))
	for range 2 {
		resp, err := c.Chat(callContext(t), req)
		if err != nil {
			t.Fatal(err)
		}
		checkAnswer(t, resp, tg)
		read, _ := resp.Usage.CacheRead()
		write, _ := resp.Usage.CacheWrite()
		t.Logf("cache read %d, write %d", read, write)
	}
}

func checkAnswer(t *testing.T, resp *litellm.Response, tg target) {
	t.Helper()
	if resp.FinishReason != litellm.FinishReasonStop || strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("finish %q, text %q", resp.FinishReason, resp.Text())
	}
	checkUsage(t, resp.Usage, tg)
}

func checkUsage(t *testing.T, u litellm.Usage, tg target) {
	t.Helper()
	counts, _ := json.Marshal(u)
	if u.InputTokens == nil || u.OutputTokens == nil {
		t.Fatalf("usage %s lacks input or output tokens", counts)
	}
	if tg.pricedUsage && (u.CacheReadTokens == nil || u.CacheWriteTokens == nil) {
		t.Fatalf("usage %s lacks cache counts", counts)
	}
}

// callContext bounds one vendor call, so a stalled vendor fails its scenario.
func callContext(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	t.Cleanup(cancel)
	return ctx
}

func request(tg target, messages ...litellm.Message) litellm.Request {
	// Reasoning models spend part of the cap thinking.
	return litellm.Request{Model: tg.model, Messages: messages, MaxTokens: new(2048)}
}

func userText(text string) litellm.Message {
	return litellm.Message{Role: litellm.RoleUser, Blocks: []litellm.Block{litellm.TextBlock{Text: text}}}
}

func newClient(t *testing.T, name string, cfg Config) *litellm.Client {
	t.Helper()
	p, err := New(name, cfg)
	if err != nil {
		t.Fatal(err)
	}
	c, err := litellm.New(p)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// recorder sends requests and keeps the bodies of successful responses, in
// order. Request headers, and so keys, are never kept.
type recorder struct {
	bodies [][]byte
	stream []bool
}

// liveHTTP retries the temporary errors vendors shed load with.
var liveHTTP = retry.NewHTTPClient(nil, &retry.Policy{
	MaxAttempts:       5,
	InitialDelay:      2 * time.Second,
	MaxDelay:          20 * time.Second,
	Multiplier:        2,
	Jitter:            true,
	RespectRetryAfter: true,
	MaxRetryAfter:     time.Minute,
})

func (r *recorder) Do(req *http.Request) (*http.Response, error) {
	resp, err := liveHTTP.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return resp, err
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, err
	}
	r.bodies = append(r.bodies, body)
	r.stream = append(r.stream, strings.Contains(resp.Header.Get("Content-Type"), "event-stream"))
	resp.Body = io.NopCloser(bytes.NewReader(body))
	return resp, nil
}

func (r *recorder) save(t *testing.T, dir string) {
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i, body := range r.bodies {
		ext := ".json"
		if r.stream[i] {
			ext = ".sse"
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%02d%s", i+1, ext)), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// replayServer answers each request with the next recorded response.
func replayServer(t *testing.T, files []string) *httptest.Server {
	var (
		mu   sync.Mutex
		next int
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if next == len(files) {
			t.Errorf("request %d has no recorded response", next+1)
			http.Error(w, "no recorded response", http.StatusInternalServerError)
			return
		}
		body, err := os.ReadFile(files[next])
		if err != nil {
			t.Error(err)
			return
		}
		contentType := "application/json"
		if strings.HasSuffix(files[next], ".sse") {
			contentType = "text/event-stream"
		}
		next++
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write(body)
	}))
	t.Cleanup(func() {
		server.Close()
		if next != len(files) {
			t.Errorf("%d of %d recorded responses used", next, len(files))
		}
	})
	return server
}
