package word2vec

import "sort"

// Vocab holds the word-to-index mapping and frequency counts.
type Vocab struct {
	Words     []string       // index → word
	Index     map[string]int // word → index
	Frequency map[string]int // word → count
	MinCount  int
}

// NewVocab creates an empty vocabulary.
func NewVocab(minCount int) *Vocab {
	return &Vocab{
		Words:     nil,
		Index:     make(map[string]int),
		Frequency: make(map[string]int),
		MinCount:  minCount,
	}
}

// BuildVocab scans sentences, counts word frequencies, filters by minCount, and assigns indices.
// Words are sorted by frequency (descending) then alphabetically for deterministic ordering.
func BuildVocab(sentences [][]string, minCount int) *Vocab {
	v := NewVocab(minCount)

	// Count frequencies
	for _, sent := range sentences {
		for _, w := range sent {
			v.Frequency[w]++
		}
	}

	// Collect words meeting minCount threshold
	var candidates []string
	for w, c := range v.Frequency {
		if c >= minCount {
			candidates = append(candidates, w)
		}
	}

	// Sort: by frequency descending, then alphabetically for stability
	sort.Slice(candidates, func(i, j int) bool {
		fi, fj := v.Frequency[candidates[i]], v.Frequency[candidates[j]]
		if fi != fj {
			return fi > fj
		}
		return candidates[i] < candidates[j]
	})

	v.Words = candidates
	for i, w := range v.Words {
		v.Index[w] = i
	}

	return v
}

// ExtendVocab adds new words from new sentences to the existing vocabulary.
// Existing indices are preserved; new words are appended. Frequency counts are updated for all words.
func (v *Vocab) ExtendVocab(sentences [][]string) {
	for _, sent := range sentences {
		for _, w := range sent {
			v.Frequency[w]++
		}
	}

	// Add new words that meet minCount
	for w, c := range v.Frequency {
		if c >= v.MinCount {
			if _, exists := v.Index[w]; !exists {
				v.Index[w] = len(v.Words)
				v.Words = append(v.Words, w)
			}
		}
	}
}

// Size returns the number of words in the vocabulary.
func (v *Vocab) Size() int {
	return len(v.Words)
}

// Contains checks if a word is in the vocabulary.
func (v *Vocab) Contains(word string) bool {
	_, ok := v.Index[word]
	return ok
}
