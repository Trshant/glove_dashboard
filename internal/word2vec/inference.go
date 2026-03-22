package word2vec

import (
	"math"
	"sort"
)

// WordScore holds a word and its similarity score.
type WordScore struct {
	Word  string
	Score float64
}

// MostSimilar returns the top N words most similar to the given word by cosine similarity.
func (m *Model) MostSimilar(word string, topN int) []WordScore {
	idx, ok := m.Vocab.Index[word]
	if !ok {
		return nil
	}

	// Normalize the target vector
	target := m.SynIn[idx]
	var targetNorm float64
	for _, v := range target {
		targetNorm += float64(v) * float64(v)
	}
	targetNorm = math.Sqrt(targetNorm)
	if targetNorm == 0 {
		return nil
	}

	type scored struct {
		idx   int
		score float64
	}

	scores := make([]scored, 0, m.Vocab.Size())
	for i := 0; i < m.Vocab.Size(); i++ {
		if i == idx {
			continue
		}
		vec := m.SynIn[i]
		var dot, norm float64
		for j, v := range vec {
			dot += float64(target[j]) * float64(v)
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			continue
		}
		sim := dot / (targetNorm * norm)
		scores = append(scores, scored{i, sim})
	}

	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	if topN > len(scores) {
		topN = len(scores)
	}

	result := make([]WordScore, topN)
	for i := 0; i < topN; i++ {
		result[i] = WordScore{
			Word:  m.Vocab.Words[scores[i].idx],
			Score: scores[i].score,
		}
	}
	return result
}

// GetSimilarWords aggregates similarity scores from multiple seed words,
// applying negative weighting for negative words, then returns top results.
// This matches the Python glove.get_similar_words() behavior.
func (m *Model) GetSimilarWords(words []string, negWords []string, count int, filterNegatives bool) []WordScore {
	// Aggregate scores using a map (like Python Counter)
	scoreMap := make(map[string]float64)

	// Positive words
	for _, w := range words {
		similar := m.MostSimilar(w, count)
		for _, ws := range similar {
			scoreMap[ws.Word] += ws.Score
		}
	}

	// Negative words (weighted at -1.25x)
	for _, w := range negWords {
		similar := m.MostSimilar(w, count)
		for _, ws := range similar {
			scoreMap[ws.Word] += ws.Score * -1.25
		}
	}

	// Normalize by total word count
	totalWords := len(words) + len(negWords)
	if totalWords > 0 {
		for k := range scoreMap {
			scoreMap[k] /= float64(totalWords)
		}
	}

	// Convert to sorted slice
	var results []WordScore
	for word, score := range scoreMap {
		if filterNegatives && score < 0 {
			continue
		}
		results = append(results, WordScore{Word: word, Score: score})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if count > len(results) {
		count = len(results)
	}
	return results[:count]
}
