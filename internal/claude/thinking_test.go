package claude

import (
	"reflect"
	"testing"

	"github.com/voocel/litellm"
)

func TestThinking(t *testing.T) {
	budget := 2048
	for _, test := range []struct {
		name string
		in   *litellm.Thinking
		want *ThinkingConfig
	}{
		{name: "nil", in: nil, want: nil},
		{name: "disabled", in: &litellm.Thinking{Disabled: true}, want: &ThinkingConfig{Type: "disabled"}},
		{name: "enabled", in: &litellm.Thinking{}, want: &ThinkingConfig{Type: "adaptive"}},
		{name: "effort", in: &litellm.Thinking{Effort: "max"}, want: &ThinkingConfig{Type: "adaptive"}},
		{name: "budget", in: &litellm.Thinking{BudgetTokens: &budget}, want: &ThinkingConfig{Type: "enabled", BudgetTokens: &budget}},
		{name: "include output", in: &litellm.Thinking{IncludeOutput: true}, want: &ThinkingConfig{Type: "adaptive", Display: "summarized"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := Thinking(test.in); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("Thinking = %+v, want %+v", got, test.want)
			}
		})
	}
}
