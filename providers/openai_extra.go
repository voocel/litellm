package providers

import "fmt"

var openAIChatExtraKeys = []string{
	"frequency_penalty",
	"presence_penalty",
	"logit_bias",
	"n",
	"logprobs",
	"top_logprobs",
	"store",
	"prompt_cache_key",
	"prompt_cache_retention",
	"prediction",
	"metadata",
	"modalities",
	"service_tier",
	"user",
	"seed",
}

func validateOpenAIChatExtra(extra map[string]any) error {
	allowed := make(map[string]struct{}, len(openAIChatExtraKeys))
	for _, key := range openAIChatExtraKeys {
		allowed[key] = struct{}{}
	}
	for key := range extra {
		if _, ok := allowed[key]; !ok {
			return fmt.Errorf("openai: unsupported extra parameter %q", key)
		}
	}
	return nil
}

func applyOpenAIChatExtra(req *openaiRequest, extra map[string]any) error {
	if len(extra) == 0 {
		return nil
	}

	for key, value := range extra {
		switch key {
		case "frequency_penalty":
			v, err := extraFloat64(key, value)
			if err != nil {
				return err
			}
			req.FrequencyPenalty = &v
		case "presence_penalty":
			v, err := extraFloat64(key, value)
			if err != nil {
				return err
			}
			req.PresencePenalty = &v
		case "logit_bias":
			v, err := extraIntMap(key, value)
			if err != nil {
				return err
			}
			req.LogitBias = v
		case "n":
			v, err := extraInt(key, value)
			if err != nil {
				return err
			}
			req.N = &v
		case "logprobs":
			v, err := extraBool(key, value)
			if err != nil {
				return err
			}
			req.Logprobs = &v
		case "top_logprobs":
			v, err := extraInt(key, value)
			if err != nil {
				return err
			}
			req.TopLogprobs = &v
		case "store":
			v, err := extraBool(key, value)
			if err != nil {
				return err
			}
			req.Store = &v
		case "prompt_cache_key":
			v, err := extraString(key, value)
			if err != nil {
				return err
			}
			req.PromptCacheKey = v
		case "prompt_cache_retention":
			v, err := extraString(key, value)
			if err != nil {
				return err
			}
			req.PromptCacheRetention = v
		case "prediction":
			v, err := extraPrediction(key, value)
			if err != nil {
				return err
			}
			req.Prediction = v
		case "metadata":
			v, err := extraStringMap(key, value)
			if err != nil {
				return err
			}
			req.Metadata = v
		case "modalities":
			v, err := extraStringSlice(key, value)
			if err != nil {
				return err
			}
			req.Modalities = v
		case "service_tier":
			v, err := extraString(key, value)
			if err != nil {
				return err
			}
			req.ServiceTier = v
		case "user":
			v, err := extraString(key, value)
			if err != nil {
				return err
			}
			req.User = v
		case "seed":
			v, err := extraInt(key, value)
			if err != nil {
				return err
			}
			req.Seed = &v
		}
	}
	return nil
}

func extraString(key string, value any) (string, error) {
	v, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("openai: extra %q must be string", key)
	}
	return v, nil
}

func extraBool(key string, value any) (bool, error) {
	v, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("openai: extra %q must be bool", key)
	}
	return v, nil
}

func extraFloat64(key string, value any) (float64, error) {
	switch v := value.(type) {
	case float64:
		return v, nil
	case float32:
		return float64(v), nil
	case int:
		return float64(v), nil
	case int64:
		return float64(v), nil
	default:
		return 0, fmt.Errorf("openai: extra %q must be number", key)
	}
}

func extraInt(key string, value any) (int, error) {
	switch v := value.(type) {
	case int:
		return v, nil
	case int64:
		return int(v), nil
	case float64:
		if v == float64(int(v)) {
			return int(v), nil
		}
	}
	return 0, fmt.Errorf("openai: extra %q must be integer", key)
}

func extraStringMap(key string, value any) (map[string]string, error) {
	switch v := value.(type) {
	case map[string]string:
		return v, nil
	case map[string]any:
		result := make(map[string]string, len(v))
		for mapKey, mapValue := range v {
			stringValue, ok := mapValue.(string)
			if !ok {
				return nil, fmt.Errorf("openai: extra %q values must be strings", key)
			}
			result[mapKey] = stringValue
		}
		return result, nil
	default:
		return nil, fmt.Errorf("openai: extra %q must be map[string]string", key)
	}
}

func extraIntMap(key string, value any) (map[string]int, error) {
	switch v := value.(type) {
	case map[string]int:
		return v, nil
	case map[string]any:
		result := make(map[string]int, len(v))
		for mapKey, mapValue := range v {
			intValue, err := extraInt(key+"."+mapKey, mapValue)
			if err != nil {
				return nil, err
			}
			result[mapKey] = intValue
		}
		return result, nil
	default:
		return nil, fmt.Errorf("openai: extra %q must be map[string]int", key)
	}
}

func extraStringSlice(key string, value any) ([]string, error) {
	switch v := value.(type) {
	case []string:
		return v, nil
	case []any:
		result := make([]string, len(v))
		for i, item := range v {
			stringItem, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf("openai: extra %q item %d must be string", key, i)
			}
			result[i] = stringItem
		}
		return result, nil
	default:
		return nil, fmt.Errorf("openai: extra %q must be []string", key)
	}
}

func extraPrediction(key string, value any) (*openaiPrediction, error) {
	switch v := value.(type) {
	case openaiPrediction:
		return &v, nil
	case *openaiPrediction:
		return v, nil
	case map[string]any:
		prediction := &openaiPrediction{}
		typeValue, ok := v["type"].(string)
		if !ok || typeValue == "" {
			return nil, fmt.Errorf("openai: extra %q.type is required", key)
		}
		contentValue, ok := v["content"].(string)
		if !ok {
			return nil, fmt.Errorf("openai: extra %q.content must be string", key)
		}
		prediction.Type = typeValue
		prediction.Content = contentValue
		return prediction, nil
	default:
		return nil, fmt.Errorf("openai: extra %q must be prediction object", key)
	}
}
