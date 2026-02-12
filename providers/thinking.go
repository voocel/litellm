package providers

import (
	"fmt"
	"strings"
)

func normalizeThinking(req *Request) *ThinkingConfig {
	if req == nil || req.Thinking == nil {
		return nil
	}

	thinkingType := strings.TrimSpace(req.Thinking.Type)
	if thinkingType == "" {
		thinkingType = "enabled"
	}

	return &ThinkingConfig{
		Type:         thinkingType,
		Level:        req.Thinking.Level,
		BudgetTokens: req.Thinking.BudgetTokens,
	}
}

func isThinkingDisabled(req *Request) bool {
	if req == nil || req.Thinking == nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(req.Thinking.Type), "disabled")
}

func validateThinking(thinking *ThinkingConfig) error {
	if thinking == nil {
		return nil
	}
	thinkingType := strings.TrimSpace(thinking.Type)
	if thinkingType == "" {
		return nil
	}
	if strings.EqualFold(thinkingType, "enabled") || strings.EqualFold(thinkingType, "disabled") {
		return nil
	}
	return fmt.Errorf("thinking type must be enabled or disabled")
}
