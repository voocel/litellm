// Package tokenusage provides arithmetic for optional wire token counts.
package tokenusage

// Sum returns a total only when every component is known.
func Sum(counts ...*int) *int {
	total := 0
	for _, count := range counts {
		if count == nil {
			return nil
		}
		total += *count
	}
	return &total
}

// AddDetails adds optional, additive protocol details to a required base count.
// Protocol adapters use this only where omitted detail fields mean no increment.
func AddDetails(base *int, details ...*int) *int {
	if base == nil {
		return nil
	}
	total := *base
	for _, detail := range details {
		if detail != nil {
			total += *detail
		}
	}
	return &total
}
