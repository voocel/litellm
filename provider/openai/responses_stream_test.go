package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/voocel/litellm"
)

// sseBody encodes data payloads as a stream without event names, so the type comes
// from the payload.
func sseBody(payloads ...string) *http.Response {
	var body strings.Builder
	for _, p := range payloads {
		body.WriteString("data: " + p + "\n\n")
	}
	return streamResponse(body.String())
}

func readAll(stream litellm.Stream) ([]litellm.Event, error) {
	var events []litellm.Event
	for {
		event, err := stream.Next()
		if errors.Is(err, io.EOF) {
			return events, nil
		}
		if err != nil {
			return events, err
		}
		events = append(events, event)
	}
}

// Blocks are numbered densely in order of first appearance, whatever the
// output index; unmapped items take no index.
func TestResponsesStreamEvents(t *testing.T) {
	const searching = `{"type":"response.web_search_call.searching","sequence_number":7,"output_index":2}`
	events, err := readAll(newResponsesStream(sseBody(
		`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"function_call","call_id":"call_a","name":"a"}}`,
		`{"type":"response.output_item.added","sequence_number":2,"output_index":1,"item":{"type":"function_call","call_id":"call_b","name":"b"}}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":3,"output_index":1,"delta":"{\"b\":1}"}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":3,"output_index":1,"delta":"{\"b\":1}"}`,
		`{"type":"response.function_call_arguments.done","sequence_number":4,"output_index":1,"arguments":"{\"b\":1}"}`,
		`{"type":"response.function_call_arguments.done","sequence_number":5,"output_index":0,"arguments":"{\"a\":1}"}`,
		`{"type":"response.output_item.done","sequence_number":6,"output_index":1,"item":{"type":"function_call","call_id":"call_b","name":"b"}}`,
		searching,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":8,"output_index":3,"summary_index":0,"delta":"x"}`,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":9,"output_index":3,"summary_index":1,"delta":"y"}`,
		`{"type":"response.completed","sequence_number":10,"response":{"status":"completed","usage":{"input_tokens":1}}}`,
	), "m"))
	if err != nil {
		t.Fatal(err)
	}
	want := []litellm.Event{
		litellm.BlockStart{Index: 0, Block: litellm.ToolUseBlock{ID: "call_a", Name: "a"}},
		litellm.BlockStart{Index: 1, Block: litellm.ToolUseBlock{ID: "call_b", Name: "b"}},
		litellm.ToolUseDelta{Index: 1, Arguments: `{"b":1}`},
		litellm.ToolUseDelta{Index: 0, Arguments: `{"a":1}`},
		litellm.BlockEnd{Index: 1, Block: litellm.ToolUseBlock{ID: "call_b", Name: "b"}},
		litellm.ProviderEvent{Name: "response.web_search_call.searching", Raw: json.RawMessage(searching)},
		litellm.BlockStart{Index: 2, Block: litellm.ReasoningBlock{Summary: true}},
		litellm.ReasoningDelta{Index: 2, Text: "x"},
		litellm.ReasoningDelta{Index: 2, Text: "\ny"},
		litellm.UsageEvent{Usage: litellm.Usage{InputTokens: new(1)}},
		litellm.BlockEnd{Index: 0},
		litellm.BlockEnd{Index: 2},
		litellm.DoneEvent{FinishReason: litellm.FinishReasonToolCall, FinishReasonRaw: "completed", Provider: "openai", Model: "m"},
	}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events:\n%#v\nwant:\n%#v", events, want)
	}
}

func TestResponsesStreamTermination(t *testing.T) {
	tests := []struct {
		name     string
		payloads []string
		done     litellm.DoneEvent
		code     string
		wantErr  string
	}{{
		name:     "incomplete",
		payloads: []string{`{"type":"response.incomplete","response":{"model":"gpt-5.1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`},
		done:     litellm.DoneEvent{FinishReason: litellm.FinishReasonLength, FinishReasonRaw: "max_output_tokens", Provider: "openai", Model: "gpt-5.1"},
	}, {
		name: "refusal",
		payloads: []string{
			`{"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"refusal","refusal":""}}`,
			`{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"no"}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		},
		done: litellm.DoneEvent{FinishReason: litellm.FinishReasonSafety, FinishReasonRaw: "completed", Provider: "openai", Model: "m"},
	}, {
		name:     "failed",
		payloads: []string{`{"type":"response.failed","response":{"status":"failed","error":{"code":"server_error","message":"boom"}}}`},
		code:     "server_error", wantErr: "response failed: boom",
	}, {
		name:     "flat error",
		payloads: []string{`{"type":"error","code":"rate_limit_exceeded","message":"slow down"}`},
		code:     "rate_limit_exceeded", wantErr: "stream error: slow down",
	}, {
		name:     "nested error",
		payloads: []string{`{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}`},
		code:     "invalid_request_error", wantErr: "stream error: bad input",
	}, {
		name:     "EOF before completed",
		payloads: []string{`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"par"}`},
		wantErr:  "stream ended before response.completed",
	}, {
		name:     "missing type",
		payloads: []string{`{"delta":"x"}`},
		wantErr:  "stream event missing type",
	}}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := newResponsesStream(sseBody(test.payloads...), "m")
			events, err := readAll(stream)
			if test.wantErr != "" {
				var e *litellm.Error
				if !errors.As(err, &e) || e.Code != test.code || !strings.Contains(e.Message, test.wantErr) {
					t.Fatalf("err = %#v, want code %q message %q", err, test.code, test.wantErr)
				}
				if _, err := stream.Next(); !errors.Is(err, io.EOF) {
					t.Fatalf("Next after failure = %v, want EOF", err)
				}
				return
			}
			if err != nil || len(events) == 0 || events[len(events)-1] != test.done {
				t.Fatalf("events = %#v, err = %v", events, err)
			}
		})
	}
}

func TestResponsesStreamFinalMessageState(t *testing.T) {
	const citation = `{"type":"url_citation","url":"https://example.com"}`
	for _, phases := range []struct{ initial, final, want string }{
		{"", "final_answer", "final_answer"},
		{"commentary", "final_answer", "final_answer"},
		{"commentary", "", "commentary"},
	} {
		t.Run(phases.initial+" to "+phases.final, func(t *testing.T) {
			finalMessage := `{"id":"msg_1","type":"message","role":"assistant","phase":"` + phases.final + `","content":[{"type":"output_text","text":"a","annotations":[],"logprobs":[]},{"type":"output_text","text":"b","annotations":[` + citation + `],"logprobs":[]}]}`
			response := `{"status":"completed","model":"m","output":[` + strings.Replace(finalMessage, `"phase":"`+phases.final+`"`, `"phase":"`+phases.want+`"`, 1) + `]}`
			stream := newResponsesStream(sseBody(
				`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","phase":"`+phases.initial+`"}}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"a"}`,
				`{"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"a","annotations":[],"logprobs":[]}}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":1,"delta":"b"}`,
				`{"type":"response.content_part.done","output_index":0,"content_index":1,"part":{"type":"output_text","text":"b","annotations":[`+citation+`],"logprobs":[]}}`,
				`{"type":"response.output_item.done","output_index":0,"item":`+finalMessage+`}`,
				`{"type":"response.completed","response":`+response+`}`,
			), "m")
			defer stream.Close()
			var ends []litellm.Block
			got, err := litellm.Handle(stream, func(event litellm.Event) error {
				if end, ok := event.(litellm.BlockEnd); ok {
					ends = append(ends, end.Block)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			var parsed responsesResponse
			if err := json.Unmarshal([]byte(response), &parsed); err != nil {
				t.Fatal(err)
			}
			want := convertResponsesResponse(&parsed, "m")
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(ends, want.Blocks) {
				t.Fatalf("stream = %+v, ends = %+v, want = %+v", got, ends, want)
			}
			// The completed phase must survive the next request as well.
			body, err := buildResponsesRequest(&litellm.Request{Model: "m", Messages: []litellm.Message{litellm.Assistant(got.Blocks...)}}, false)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Count(string(body), `"phase":"`+phases.want+`"`) != 2 {
				t.Fatalf("replay lost phase: %s", body)
			}
		})
	}
}

func TestResponsesStreamContentMetadataWithoutItemDone(t *testing.T) {
	stream := newResponsesStream(sseBody(
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","phase":"commentary"}}`,
		`{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"partial"}`,
		`{"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"partial","annotations":[],"logprobs":[{"token":"partial"}]}}`,
		`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
	), "m")
	defer stream.Close()
	resp, err := litellm.Collect(stream)
	if err != nil {
		t.Fatal(err)
	}
	block := resp.Blocks[0].(litellm.TextBlock)
	if block.Text != "partial" || string(block.Logprobs) != `[{"token":"partial"}]` || resp.FinishReason != litellm.FinishReasonLength {
		t.Fatalf("response = %+v, block = %+v", resp, block)
	}
	assertJSON(t, block.State.Data, `{"id":"msg_1","phase":"commentary"}`)
}
