package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

// Official Chat Completions metadata lives on the choice and message,
// unlike Responses metadata, which belongs to a content part.
func TestChatMetadataMatchesStream(t *testing.T) {
	const citation = `{"type":"url_citation","url_citation":{"url":"https://example.com","title":"Source","start_index":0,"end_index":5}}`
	const first = `{"token":"Hel","logprob":-0.125,"bytes":[72,101,108],"top_logprobs":[{"token":"Hi","logprob":-2,"bytes":[72,105]}]}`
	const second = `{"token":"lo","logprob":-0.25,"bytes":[108,111],"top_logprobs":[]}`
	for _, field := range []string{"content", "refusal"} {
		t.Run(field, func(t *testing.T) {
			logprobs := `{"content":[` + first + `,` + second + `],"refusal":null}`
			if field == "refusal" {
				logprobs = `{"content":null,"refusal":[` + first + `,` + second + `]}`
			}
			reply := `{"model":"m","choices":[{"message":{"role":"assistant","` + field + `":"Hello","annotations":[` + citation + `]},"logprobs":` + logprobs + `,"finish_reason":"stop"}]}`
			chunks := []string{
				`{"model":"m","choices":[{"delta":{"role":"assistant"},"logprobs":null}]}`,
				`{"choices":[{"delta":{"` + field + `":"Hel"},"logprobs":{"` + field + `":[` + first + `]}}]}`,
				`{"choices":[{"delta":{"` + field + `":"lo"},"logprobs":{"` + field + `":[` + second + `]}}]}`,
				`{"choices":[{"delta":{"annotations":[` + citation + `]},"finish_reason":"stop","logprobs":null}]}`,
				`[DONE]`,
			}
			for _, streaming := range []bool{false, true} {
				body := reply
				if streaming {
					body = "data: " + strings.Join(chunks, "\n\ndata: ") + "\n\n"
				}
				p, _ := testProvider(t, Config{}, body)
				client, err := litellm.New(p)
				if err != nil {
					t.Fatal(err)
				}
				req := litellm.Request{Model: "m", Messages: []litellm.Message{litellm.UserText("hi")}, ProviderOptions: providerOptions(t, map[string]any{ProviderOptionLogprobs: true})}
				var resp *litellm.Response
				if streaming {
					stream, err := client.Stream(context.Background(), req)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { stream.Close() })
					ends := 0
					resp, err = litellm.Handle(stream, func(event litellm.Event) error {
						if end, ok := event.(litellm.BlockEnd); ok {
							ends++
							assertJSON(t, end.Block.(litellm.TextBlock).Logprobs, logprobs)
						}
						return nil
					})
					if err != nil || ends != 1 {
						t.Fatalf("stream err = %v, ends = %d", err, ends)
					}
				} else {
					resp, err = client.Chat(context.Background(), req)
					if err != nil {
						t.Fatal(err)
					}
				}
				if len(resp.Blocks) != 1 || resp.Text() != "Hello" {
					t.Fatalf("streaming=%v: response = %+v", streaming, resp)
				}
				block := resp.Blocks[0].(litellm.TextBlock)
				assertJSON(t, block.Logprobs, logprobs)
				if len(block.Annotations) != 1 || block.Annotations[0].URL != "https://example.com" || block.Annotations[0].Text != "Source" {
					t.Fatalf("annotations = %+v", block.Annotations)
				}
				assertJSON(t, block.Annotations[0].Extra, citation)
				if field == "refusal" && resp.FinishReason != litellm.FinishReasonSafety {
					t.Fatalf("refusal finish = %q", resp.FinishReason)
				}
			}
		})
	}
}

func TestChatNullLogprobs(t *testing.T) {
	p, _ := testProvider(t, Config{}, `{"choices":[{"message":{"content":"ok"},"logprobs":null,"finish_reason":"stop"}]}`)
	resp, err := p.Chat(context.Background(), &litellm.Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if block := resp.Blocks[0].(litellm.TextBlock); block.Logprobs != nil {
		t.Fatalf("null logprobs = %s", block.Logprobs)
	}
}

// Non-streaming metadata is kept verbatim, including unknown token fields.
func TestChatPreservesRawLogprobs(t *testing.T) {
	const raw = `{"content":[{"token":"x","logprob":-0.1234567890123456789,"bytes":[120],"extra":9007199254740993}],"refusal":null}`
	p, _ := testProvider(t, Config{}, `{"choices":[{"message":{"content":"x"},"logprobs":`+raw+`}]}`)
	resp, err := p.Chat(context.Background(), &litellm.Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Blocks[0].(litellm.TextBlock).Logprobs; string(got) != raw || !json.Valid(got) {
		t.Fatalf("logprobs = %s", got)
	}
}
