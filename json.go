package litellm

import (
	"encoding/json"
	"fmt"
)

// A Block encodes as a JSON object whose "type" names its kind: "text",
// "image", "reasoning", "tool_use", "tool_result" or "tool_reference". A
// Message and its blocks therefore round-trip through JSON, provider state
// included, so history can be stored and sent as JSON.

func (b TextBlock) MarshalJSON() ([]byte, error) {
	type plain TextBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"text", plain(b)})
}

func (b ImageBlock) MarshalJSON() ([]byte, error) {
	type plain ImageBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"image", plain(b)})
}

func (b ReasoningBlock) MarshalJSON() ([]byte, error) {
	type plain ReasoningBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"reasoning", plain(b)})
}

func (b ToolUseBlock) MarshalJSON() ([]byte, error) {
	type plain ToolUseBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"tool_use", plain(b)})
}

func (b ToolResultBlock) MarshalJSON() ([]byte, error) {
	type plain ToolResultBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"tool_result", plain(b)})
}

func (b ToolReferenceBlock) MarshalJSON() ([]byte, error) {
	type plain ToolReferenceBlock
	return json.Marshal(struct {
		Type string `json:"type"`
		plain
	}{"tool_reference", plain(b)})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	type plain Message
	var v struct {
		plain
		Blocks []json.RawMessage `json:"blocks"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	blocks, err := UnmarshalBlocks(v.Blocks)
	if err != nil {
		return err
	}
	*m = Message(v.plain)
	m.Blocks = blocks
	return nil
}

func (b *ToolResultBlock) UnmarshalJSON(data []byte) error {
	type plain ToolResultBlock
	var v struct {
		plain
		Content []json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	content, err := UnmarshalBlocks(v.Content)
	if err != nil {
		return err
	}
	*b = ToolResultBlock(v.plain)
	b.Content = content
	return nil
}

// UnmarshalBlocks decodes blocks encoded as JSON, each by its "type".
func UnmarshalBlocks(raws []json.RawMessage) ([]Block, error) {
	if raws == nil {
		return nil, nil
	}
	blocks := make([]Block, len(raws))
	for i, raw := range raws {
		block, err := UnmarshalBlock(raw)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
		blocks[i] = block
	}
	return blocks, nil
}

// UnmarshalBlock decodes a block encoded as JSON by its "type".
func UnmarshalBlock(raw json.RawMessage) (Block, error) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, err
	}
	switch head.Type {
	case "text":
		return decode[TextBlock](raw)
	case "image":
		return decode[ImageBlock](raw)
	case "reasoning":
		return decode[ReasoningBlock](raw)
	case "tool_use":
		return decode[ToolUseBlock](raw)
	case "tool_result":
		return decode[ToolResultBlock](raw)
	case "tool_reference":
		return decode[ToolReferenceBlock](raw)
	default:
		return nil, fmt.Errorf("unknown block type %q", head.Type)
	}
}

func decode[B Block](raw json.RawMessage) (Block, error) {
	var b B
	err := json.Unmarshal(raw, &b)
	return b, err
}

// MarshalJSON writes the schema as the JSON document it is.
func (s Schema) MarshalJSON() ([]byte, error) {
	if len(s) == 0 {
		return []byte("null"), nil
	}
	return json.RawMessage(s).MarshalJSON()
}

// UnmarshalJSON keeps a copy of the JSON document.
func (s *Schema) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		*s = nil
		return nil
	}
	*s = Schema(cloneBytes(data))
	return nil
}
