package word2vec

import "testing"

func TestBuildVocab(t *testing.T) {
	sentences := [][]string{
		{"the", "cat", "sat", "on", "the", "mat"},
		{"the", "dog", "sat", "on", "the", "log"},
		{"the", "cat", "and", "dog"},
	}

	v := BuildVocab(sentences, 2)

	// "the" appears 5 times, "cat" 2, "sat" 2, "on" 2, "dog" 2
	// "mat", "log", "and" appear only once — should be excluded with minCount=2
	if !v.Contains("the") {
		t.Error("expected 'the' in vocab")
	}
	if !v.Contains("cat") {
		t.Error("expected 'cat' in vocab")
	}
	if v.Contains("mat") {
		t.Error("'mat' should not be in vocab (count=1 < minCount=2)")
	}
	if v.Contains("and") {
		t.Error("'and' should not be in vocab (count=1 < minCount=2)")
	}

	// Check frequency
	if v.Frequency["the"] != 5 {
		t.Errorf("expected freq('the')=5, got %d", v.Frequency["the"])
	}
}

func TestExtendVocab(t *testing.T) {
	sentences := [][]string{
		{"the", "cat", "sat", "on", "the", "mat"},
		{"the", "dog", "sat", "on", "the", "log"},
	}

	v := BuildVocab(sentences, 2)
	origSize := v.Size()
	catIdx := v.Index["cat"]

	// Add new sentences with a new word that now meets threshold
	newSentences := [][]string{
		{"the", "mat", "is", "new"},
		{"the", "mat", "is", "old"},
	}
	v.ExtendVocab(newSentences)

	// "mat" now has count 3 (1+2), should be added
	if !v.Contains("mat") {
		t.Error("expected 'mat' in vocab after extension")
	}

	// Original indices should be preserved
	if v.Index["cat"] != catIdx {
		t.Errorf("expected cat index %d preserved, got %d", catIdx, v.Index["cat"])
	}

	// Size should have increased
	if v.Size() <= origSize {
		t.Error("expected vocab size to increase after extension")
	}
}
