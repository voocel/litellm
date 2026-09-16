package litellm

import "fmt"

type contentAddress struct {
	output, content int
}

type contentState struct {
	index  int
	closed bool
}

func addressOf(output, content *int) contentAddress {
	address := contentAddress{output: -1, content: -1}
	if output != nil {
		address.output = *output
	}
	if content != nil {
		address.content = *content
	}
	return address
}

func validateContentAddress(output, content *int) error {
	if output == nil && content == nil {
		return fmt.Errorf("content boundary requires a block coordinate")
	}
	if output != nil && *output < 0 || content != nil && *content < 0 {
		return fmt.Errorf("content block coordinates cannot be negative")
	}
	return nil
}

func contentKind(block Block) (string, bool, error) {
	switch block := block.(type) {
	case TextBlock:
		return "text", false, nil
	case ReasoningBlock:
		return "reasoning", block.Summary, nil
	default:
		return "", false, fmt.Errorf("content boundary does not support block %T", block)
	}
}

func (c *EventCollector) startContent(event ContentStart) error {
	if err := validateContentAddress(event.OutputIndex, event.ContentIndex); err != nil {
		return err
	}
	kind, summary, err := contentKind(event.Block)
	if err != nil {
		return err
	}
	address := addressOf(event.OutputIndex, event.ContentIndex)
	if _, exists := c.contentStates[address]; exists {
		return fmt.Errorf("content block started more than once")
	}
	if _, exists := c.contentIndexes[address]; exists {
		return fmt.Errorf("content block started after its deltas")
	}
	// Register the boundary first: an explicit output-only address is a block,
	// unlike an output-only delta from a protocol with no block coordinates.
	c.contentStates[address] = contentState{}
	index := c.contentIndex(kind, event.OutputIndex, event.ContentIndex, summary)
	c.blocks[index] = cloneBlock(event.Block)
	c.contentStates[address] = contentState{index: index}
	return nil
}

func (c *EventCollector) endContent(event ContentEnd) error {
	if err := validateContentAddress(event.OutputIndex, event.ContentIndex); err != nil {
		return err
	}
	address := addressOf(event.OutputIndex, event.ContentIndex)
	state, exists := c.contentStates[address]
	if !exists {
		return fmt.Errorf("content block ended without a start")
	}
	if state.closed {
		return fmt.Errorf("content block ended more than once")
	}
	if event.Block != nil {
		kind, summary, err := contentKind(event.Block)
		if err != nil {
			return err
		}
		if err := c.checkContentDelta(event.OutputIndex, event.ContentIndex, kind, summary); err != nil {
			return err
		}
		currentText := contentText(c.blocks[state.index])
		if builder := c.textBuilders[state.index]; builder != nil {
			currentText = builder.String()
		}
		if currentText != contentText(event.Block) {
			return fmt.Errorf("final content snapshot differs from streamed text")
		}
		c.blocks[state.index] = cloneBlock(event.Block)
		delete(c.textBuilders, state.index)
	}
	state.closed = true
	c.contentStates[address] = state
	return nil
}

func contentText(block Block) string {
	switch block := block.(type) {
	case TextBlock:
		return block.Text
	case ReasoningBlock:
		return block.Text
	default:
		return ""
	}
}

func (c *EventCollector) checkContentDelta(output, content *int, kind string, summary bool) error {
	if output != nil && *output < 0 || content != nil && *content < 0 {
		return fmt.Errorf("content block coordinates cannot be negative")
	}
	address := addressOf(output, content)
	if state, exists := c.contentStates[address]; exists && state.closed {
		return fmt.Errorf("content delta received after block end")
	}
	index, exists := c.contentIndexes[address]
	if !exists {
		return nil
	}
	wantKind, wantSummary, err := contentKind(c.blocks[index])
	if err != nil {
		return err
	}
	if kind != wantKind || summary != wantSummary {
		return fmt.Errorf("content delta type does not match its block")
	}
	return nil
}
