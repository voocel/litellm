package wire

import (
	"cmp"
	"maps"
	"slices"

	"github.com/voocel/litellm"
)

// BlockTracker maps provider-native keys to open litellm block indexes. The zero
// value is ready to use.
type BlockTracker[K comparable] struct {
	next int
	open map[K]int
}

// Open returns the index of key's open block, first appending a BlockStart for
// block when key has none.
func (t *BlockTracker[K]) Open(events []litellm.Event, key K, block litellm.Block) ([]litellm.Event, int) {
	if index, ok := t.open[key]; ok {
		return events, index
	}
	if t.open == nil {
		t.open = make(map[K]int)
	}
	index := t.next
	t.next++
	t.open[key] = index
	return append(events, litellm.BlockStart{Index: index, Block: block}), index
}

// Index reports the index of key's open block.
func (t *BlockTracker[K]) Index(key K) (int, bool) {
	index, ok := t.open[key]
	return index, ok
}

// Close appends a BlockEnd for key's open block. final may be nil or carry
// late metadata; see litellm.BlockEnd.
func (t *BlockTracker[K]) Close(events []litellm.Event, key K, final litellm.Block) []litellm.Event {
	index, ok := t.open[key]
	if !ok {
		return events
	}
	delete(t.open, key)
	return append(events, litellm.BlockEnd{Index: index, Block: final})
}

// CloseAll appends a BlockEnd for every open block, in index order. final,
// when not nil, supplies each block's late metadata.
func (t *BlockTracker[K]) CloseAll(events []litellm.Event, final func(K) litellm.Block) []litellm.Event {
	keys := slices.SortedFunc(maps.Keys(t.open), func(a, b K) int { return cmp.Compare(t.open[a], t.open[b]) })
	for _, key := range keys {
		var block litellm.Block
		if final != nil {
			block = final(key)
		}
		events = append(events, litellm.BlockEnd{Index: t.open[key], Block: block})
	}
	clear(t.open)
	return events
}
