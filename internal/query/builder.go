package query

// BuildQuery constructs an Elasticsearch bool query from positive and negative terms.
// This matches the Python utils.list2elastic() behavior:
//   - Positive terms: bool → should[match_phrase(term)]
//   - Negative terms: bool → must[match_phrase(term)]
//   - Wrapped in outer bool → should
func BuildQuery(positive, negative []string) map[string]any {
	var shouldClauses []any

	// Positive terms: should (OR)
	if len(positive) > 0 {
		var posMatches []any
		for _, term := range positive {
			posMatches = append(posMatches, map[string]any{
				"match_phrase": map[string]any{
					"content": term,
				},
			})
		}
		shouldClauses = append(shouldClauses, map[string]any{
			"bool": map[string]any{
				"should": posMatches,
			},
		})
	}

	// Negative terms: must (AND — requires these terms to be present)
	if len(negative) > 0 {
		var negMatches []any
		for _, term := range negative {
			negMatches = append(negMatches, map[string]any{
				"match_phrase": map[string]any{
					"content": term,
				},
			})
		}
		shouldClauses = append(shouldClauses, map[string]any{
			"bool": map[string]any{
				"must": negMatches,
			},
		})
	}

	return map[string]any{
		"from":            0,
		"size":            100,
		"track_total_hits": true,
		"query": map[string]any{
			"bool": map[string]any{
				"should": shouldClauses,
			},
		},
	}
}
