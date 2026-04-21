package litellm

import (
	"context"
	"errors"
	"io"
	"testing"
)

func assertCallMeta(t *testing.T, before, after CallMeta, operation, model string, streaming bool) {
	t.Helper()

	if before.CallID == "" {
		t.Fatal("call_id is empty")
	}
	if after.CallID != before.CallID {
		t.Fatalf("after call_id = %q, want %q", after.CallID, before.CallID)
	}
	if before.Operation != operation {
		t.Fatalf("operation = %q, want %q", before.Operation, operation)
	}
	if before.Model != model {
		t.Fatalf("model = %q, want %q", before.Model, model)
	}
	if before.Streaming != streaming {
		t.Fatalf("streaming = %v, want %v", before.Streaming, streaming)
	}
	if after.Duration <= 0 {
		t.Fatalf("duration = %v, want > 0", after.Duration)
	}
}

func assertInternalErrorModel(t *testing.T, err error, model string) {
	t.Helper()

	var llmErr *LiteLLMError
	if !errors.As(err, &llmErr) {
		t.Fatalf("error type = %T, want *LiteLLMError", err)
	}
	if llmErr.Type != ErrorTypeInternal {
		t.Fatalf("error type = %q, want %q", llmErr.Type, ErrorTypeInternal)
	}
	if llmErr.Model != model {
		t.Fatalf("error model = %q, want %q", llmErr.Model, model)
	}
}

type pipelineTestProvider struct {
	name       string
	chatFunc   func(context.Context, *Request) (*Response, error)
	streamFunc func(context.Context, *Request) (StreamReader, error)
	lastReq    *Request
}

func (p *pipelineTestProvider) Name() string { return p.name }

func (p *pipelineTestProvider) Validate() error { return nil }

func (p *pipelineTestProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	p.lastReq = req
	if p.chatFunc == nil {
		return &Response{Model: req.Model, Provider: p.name}, nil
	}
	return p.chatFunc(ctx, req)
}

func (p *pipelineTestProvider) Stream(ctx context.Context, req *Request) (StreamReader, error) {
	p.lastReq = req
	if p.streamFunc == nil {
		return &pipelineTestStream{
			chunks: []*StreamChunk{{Type: ChunkTypeContent, Content: "ok", Done: true}},
		}, nil
	}
	return p.streamFunc(ctx, req)
}

type pipelineTestStream struct {
	chunks []*StreamChunk
	index  int
	closed bool
}

func (s *pipelineTestStream) Next() (*StreamChunk, error) {
	if s.index >= len(s.chunks) {
		return nil, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (s *pipelineTestStream) Close() error {
	s.closed = true
	return nil
}

type responsesPipelineTestProvider struct {
	*pipelineTestProvider
	responsesFunc       func(context.Context, *OpenAIResponsesRequest) (*Response, error)
	responsesStreamFunc func(context.Context, *OpenAIResponsesRequest) (StreamReader, error)
	lastResponsesReq    *OpenAIResponsesRequest
}

func (p *responsesPipelineTestProvider) Responses(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
	p.lastResponsesReq = req
	if p.responsesFunc == nil {
		return &Response{Model: req.Model, Provider: p.name}, nil
	}
	return p.responsesFunc(ctx, req)
}

func (p *responsesPipelineTestProvider) ResponsesStream(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
	p.lastResponsesReq = req
	if p.responsesStreamFunc == nil {
		return &pipelineTestStream{
			chunks: []*StreamChunk{{Type: ChunkTypeContent, Content: "ok", Done: true}},
		}, nil
	}
	return p.responsesStreamFunc(ctx, req)
}

func TestClientChatHooks(t *testing.T) {
	provider := &pipelineTestProvider{
		name: "hook-test",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			return &Response{
				Model:        req.Model,
				Provider:     "hook-test",
				Content:      "pong",
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	var beforeMeta CallMeta
	var afterMeta CallMeta
	var beforeCalled int
	var afterCalled int
	var afterResp *Response

	client, err := New(provider, WithHook(HookFuncs{
		BeforeRequestFunc: func(ctx context.Context, meta CallMeta) {
			beforeCalled++
			beforeMeta = meta
		},
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			afterCalled++
			afterMeta = meta
			afterResp = resp
			if err != nil {
				t.Fatalf("AfterResponse returned unexpected error: %v", err)
			}
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	resp, err := client.Chat(context.Background(), NewRequest("test-model", "ping"))
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if beforeCalled != 1 {
		t.Fatalf("BeforeRequest called %d times, want 1", beforeCalled)
	}
	if afterCalled != 1 {
		t.Fatalf("AfterResponse called %d times, want 1", afterCalled)
	}
	if resp != afterResp {
		t.Fatal("AfterResponse did not receive the final response")
	}
	assertCallMeta(t, beforeMeta, afterMeta, "chat", "test-model", false)
	if provider.lastReq == nil || provider.lastReq.MaxTokens == nil {
		t.Fatal("prepared request defaults were not applied")
	}
}

func TestClientStreamHooks(t *testing.T) {
	baseStream := &pipelineTestStream{
		chunks: []*StreamChunk{
			{Type: ChunkTypeContent, Content: "a", Done: false},
			{Type: ChunkTypeContent, Content: "b", Done: true},
		},
	}
	provider := &pipelineTestProvider{
		name: "hook-test",
		streamFunc: func(ctx context.Context, req *Request) (StreamReader, error) {
			return baseStream, nil
		},
	}

	var beforeMeta CallMeta
	var afterMeta CallMeta
	var chunkMetas []CallMeta
	var chunkContents []string
	var afterResp *Response
	var beforeCalled int
	var afterCalled int

	client, err := New(provider, WithHook(HookFuncs{
		BeforeRequestFunc: func(ctx context.Context, meta CallMeta) {
			beforeCalled++
			beforeMeta = meta
		},
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			afterCalled++
			afterMeta = meta
			afterResp = resp
			if err != nil {
				t.Fatalf("AfterResponse returned unexpected error: %v", err)
			}
		},
		OnStreamChunkFunc: func(ctx context.Context, meta CallMeta, chunk *StreamChunk) {
			chunkMetas = append(chunkMetas, meta)
			chunkContents = append(chunkContents, chunk.Content)
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	stream, err := client.Stream(context.Background(), NewRequest("stream-model", "say hi"))
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}

	resp, err := CollectStream(stream)
	if err != nil {
		t.Fatalf("CollectStream returned error: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}

	if beforeCalled != 1 {
		t.Fatalf("BeforeRequest called %d times, want 1", beforeCalled)
	}
	if afterCalled != 1 {
		t.Fatalf("AfterResponse called %d times, want 1", afterCalled)
	}
	assertCallMeta(t, beforeMeta, afterMeta, "stream", "stream-model", true)
	if afterResp != nil {
		t.Fatal("stream AfterResponse should receive nil response when stream is established")
	}
	if len(chunkContents) != 2 {
		t.Fatalf("OnStreamChunk called %d times, want 2", len(chunkContents))
	}
	if chunkContents[0] != "a" || chunkContents[1] != "b" {
		t.Fatalf("unexpected chunk contents: %#v", chunkContents)
	}
	for i, meta := range chunkMetas {
		if meta.CallID != beforeMeta.CallID {
			t.Fatalf("chunk meta %d call_id = %q, want %q", i, meta.CallID, beforeMeta.CallID)
		}
	}
	if resp.Content != "ab" {
		t.Fatalf("stream response content = %q, want %q", resp.Content, "ab")
	}
}

func TestClientChatNilResponseReturnsInternalError(t *testing.T) {
	provider := &pipelineTestProvider{
		name: "hook-test",
		chatFunc: func(ctx context.Context, req *Request) (*Response, error) {
			return nil, nil
		},
	}

	var hookErr error
	client, err := New(provider, WithHook(HookFuncs{
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			hookErr = err
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = client.Chat(context.Background(), NewRequest("test-model", "ping"))
	if err == nil {
		t.Fatal("Chat returned nil error, want internal error")
	}

	assertInternalErrorModel(t, err, "test-model")
	if hookErr == nil {
		t.Fatal("AfterResponse did not receive the internal error")
	}
}

func TestClientStreamNilStreamReturnsInternalError(t *testing.T) {
	provider := &pipelineTestProvider{
		name: "hook-test",
		streamFunc: func(ctx context.Context, req *Request) (StreamReader, error) {
			return nil, nil
		},
	}

	var hookErr error
	client, err := New(provider, WithHook(HookFuncs{
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			hookErr = err
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = client.Stream(context.Background(), NewRequest("test-model", "ping"))
	if err == nil {
		t.Fatal("Stream returned nil error, want internal error")
	}

	assertInternalErrorModel(t, err, "test-model")
	if hookErr == nil {
		t.Fatal("AfterResponse did not receive the internal error")
	}
}

func TestClientResponsesHooks(t *testing.T) {
	provider := &responsesPipelineTestProvider{
		pipelineTestProvider: &pipelineTestProvider{name: "openai"},
		responsesFunc: func(ctx context.Context, req *OpenAIResponsesRequest) (*Response, error) {
			return &Response{
				Model:        req.Model,
				Provider:     "openai",
				Content:      "response-pong",
				FinishReason: FinishReasonStop,
			}, nil
		},
	}

	var beforeMeta CallMeta
	var afterMeta CallMeta
	var beforeCalled int
	var afterCalled int
	var afterResp *Response

	client, err := New(provider, WithHook(HookFuncs{
		BeforeRequestFunc: func(ctx context.Context, meta CallMeta) {
			beforeCalled++
			beforeMeta = meta
		},
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			afterCalled++
			afterMeta = meta
			afterResp = resp
			if err != nil {
				t.Fatalf("AfterResponse returned unexpected error: %v", err)
			}
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := &OpenAIResponsesRequest{
		Model:    "gpt-5.4",
		Messages: []Message{{Role: "user", Content: "ping"}},
	}
	resp, err := client.Responses(context.Background(), req)
	if err != nil {
		t.Fatalf("Responses returned error: %v", err)
	}

	if beforeCalled != 1 {
		t.Fatalf("BeforeRequest called %d times, want 1", beforeCalled)
	}
	if afterCalled != 1 {
		t.Fatalf("AfterResponse called %d times, want 1", afterCalled)
	}
	if resp != afterResp {
		t.Fatal("AfterResponse did not receive the final response")
	}
	assertCallMeta(t, beforeMeta, afterMeta, "responses", "gpt-5.4", false)
	if provider.lastResponsesReq == nil || provider.lastResponsesReq.MaxOutputTokens == nil {
		t.Fatal("prepared responses request defaults were not applied")
	}
}

func TestClientResponsesStreamHooks(t *testing.T) {
	baseStream := &pipelineTestStream{
		chunks: []*StreamChunk{
			{Type: ChunkTypeContent, Content: "x", Done: false},
			{Type: ChunkTypeContent, Content: "y", Done: true},
		},
	}
	provider := &responsesPipelineTestProvider{
		pipelineTestProvider: &pipelineTestProvider{name: "openai"},
		responsesStreamFunc: func(ctx context.Context, req *OpenAIResponsesRequest) (StreamReader, error) {
			return baseStream, nil
		},
	}

	var beforeMeta CallMeta
	var afterMeta CallMeta
	var chunkContents []string
	var afterResp *Response
	var beforeCalled int
	var afterCalled int

	client, err := New(provider, WithHook(HookFuncs{
		BeforeRequestFunc: func(ctx context.Context, meta CallMeta) {
			beforeCalled++
			beforeMeta = meta
		},
		AfterResponseFunc: func(ctx context.Context, meta CallMeta, resp *Response, err error) {
			afterCalled++
			afterMeta = meta
			afterResp = resp
			if err != nil {
				t.Fatalf("AfterResponse returned unexpected error: %v", err)
			}
		},
		OnStreamChunkFunc: func(ctx context.Context, meta CallMeta, chunk *StreamChunk) {
			if beforeMeta.CallID != "" && meta.CallID != beforeMeta.CallID {
				t.Fatalf("unexpected chunk call_id: %q, want %q", meta.CallID, beforeMeta.CallID)
			}
			chunkContents = append(chunkContents, chunk.Content)
		},
	}))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	req := &OpenAIResponsesRequest{
		Model:    "gpt-5.4",
		Messages: []Message{{Role: "user", Content: "say hi"}},
	}
	stream, err := client.ResponsesStream(context.Background(), req)
	if err != nil {
		t.Fatalf("ResponsesStream returned error: %v", err)
	}

	resp, err := CollectStream(stream)
	if err != nil {
		t.Fatalf("CollectStream returned error: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}

	if beforeCalled != 1 {
		t.Fatalf("BeforeRequest called %d times, want 1", beforeCalled)
	}
	if afterCalled != 1 {
		t.Fatalf("AfterResponse called %d times, want 1", afterCalled)
	}
	assertCallMeta(t, beforeMeta, afterMeta, "responses_stream", "gpt-5.4", true)
	if afterResp != nil {
		t.Fatal("stream AfterResponse should receive nil response when stream is established")
	}
	if len(chunkContents) != 2 {
		t.Fatalf("OnStreamChunk called %d times, want 2", len(chunkContents))
	}
	if chunkContents[0] != "x" || chunkContents[1] != "y" {
		t.Fatalf("unexpected chunk contents: %#v", chunkContents)
	}
	if resp.Content != "xy" {
		t.Fatalf("stream response content = %q, want %q", resp.Content, "xy")
	}
	if provider.lastResponsesReq == nil || provider.lastResponsesReq.MaxOutputTokens == nil {
		t.Fatal("prepared responses request defaults were not applied")
	}
}
