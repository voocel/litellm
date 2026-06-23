package glm

import (
	"fmt"
	"strings"

	"github.com/voocel/litellm"
	"github.com/voocel/litellm/provider/compat"
)

const defaultBaseURL = "https://open.bigmodel.cn/api/paas/v4"

type Config = compat.Config

func New(cfg Config) (*compat.Provider, error) {
	return compat.New(cfg, compat.Spec{
		Name: "glm",
		Endpoint: compat.EndpointSpec{
			BaseURL: defaultBaseURL,
		},
		Auth: compat.AuthSpec{APIKeyRequired: true},
		Request: compat.RequestSpec{
			MaxStopSequences:   1,
			JSONSchemaToPrompt: true,
			ResponseFormat:     mapResponseFormat,
			Thinking:           mapThinking,
		},
		Response: compat.ResponseSpec{
			ModelFromResponse:         true,
			HasCompletionTokenDetails: true,
		},
	})
}

func Factory(cfg Config) (litellm.Provider, error) {
	return New(cfg)
}

func mapResponseFormat(format *litellm.ResponseFormat) (any, error) {
	switch format.Type {
	case litellm.ResponseFormatText:
		return nil, nil
	case litellm.ResponseFormatJSONObject, litellm.ResponseFormatJSONSchema:
		return map[string]string{"type": "json_object"}, nil
	default:
		return nil, fmt.Errorf("glm: unsupported response format %q", format.Type)
	}
}

func mapThinking(thinking *litellm.Thinking, model string) (map[string]any, error) {
	if thinking == nil || thinking.Mode == litellm.ThinkingUnspecified {
		return nil, nil
	}
	if !versionAtLeast(model, 4, 5) {
		return nil, fmt.Errorf("glm: thinking is only supported for glm-4.5 or later")
	}
	switch thinking.Mode {
	case litellm.ThinkingDisabled:
		return map[string]any{"thinking": map[string]any{"type": "disabled"}}, nil
	case litellm.ThinkingEnabled:
		body := map[string]any{"thinking": map[string]any{"type": "enabled"}}
		if thinking.Level != "" && versionAtLeast(model, 5, 2) {
			body["reasoning_effort"] = strings.ToLower(strings.TrimSpace(thinking.Level))
		}
		return body, nil
	default:
		return nil, fmt.Errorf("glm: unsupported thinking mode %d", thinking.Mode)
	}
}

func versionAtLeast(model string, major, minor int) bool {
	_, after, ok := strings.Cut(strings.ToLower(model), "glm-")
	if !ok {
		return false
	}
	var maj, min int
	fmt.Sscanf(after, "%d.%d", &maj, &min)
	return maj > major || (maj == major && min >= minor)
}
