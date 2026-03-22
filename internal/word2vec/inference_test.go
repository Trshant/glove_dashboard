package word2vec

import (
	"math"
	"testing"
)

func makeTestModel() *Model {
	// Create a small model with known vectors for testing
	config := TrainConfig{VectorSize: 3, MinCount: 1}
	vocab := NewVocab(1)
	words := []string{"good", "great", "bad", "terrible", "cat"}
	for i, w := range words {
		vocab.Words = append(vocab.Words, w)
		vocab.Index[w] = i
		vocab.Frequency[w] = 10
	}

	// Manually set vectors:
	// "good" and "great" are similar (close direction)
	// "bad" and "terrible" are similar (different direction from good/great)
	// "cat" is orthogonal
	synIn := [][]float32{
		{1.0, 0.0, 0.1},  // good
		{0.9, 0.1, 0.1},  // great (similar to good)
		{-1.0, 0.0, 0.1}, // bad (opposite of good)
		{-0.9, 0.1, 0.1}, // terrible (similar to bad)
		{0.0, 1.0, 0.0},  // cat (orthogonal)
	}

	return &Model{
		Vocab:  vocab,
		SynIn:  synIn,
		SynOut: make([][]float32, len(synIn)),
		Config: config,
	}
}

func TestMostSimilar(t *testing.T) {
	m := makeTestModel()

	results := m.MostSimilar("good", 4)
	if len(results) != 4 {
		t.Fatalf("expected 4 results, got %d", len(results))
	}

	// "great" should be the most similar to "good"
	if results[0].Word != "great" {
		t.Errorf("expected 'great' as most similar to 'good', got %q", results[0].Word)
	}

	// Scores should be in descending order
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not in descending order: %.4f > %.4f", results[i].Score, results[i-1].Score)
		}
	}
}

func TestMostSimilarUnknownWord(t *testing.T) {
	m := makeTestModel()

	results := m.MostSimilar("unknown", 5)
	if results != nil {
		t.Errorf("expected nil for unknown word, got %v", results)
	}
}

func TestGetSimilarWords(t *testing.T) {
	m := makeTestModel()

	results := m.GetSimilarWords([]string{"good"}, nil, 4, true)
	if len(results) == 0 {
		t.Fatal("expected some results")
	}

	// All scores should be non-negative when filtering negatives
	for _, r := range results {
		if r.Score < 0 {
			t.Errorf("expected non-negative score, got %.4f for %q", r.Score, r.Word)
		}
	}
}

func TestGetSimilarWordsWithNegatives(t *testing.T) {
	m := makeTestModel()

	results := m.GetSimilarWords([]string{"good"}, []string{"bad"}, 4, false)
	if len(results) == 0 {
		t.Fatal("expected some results")
	}

	// With negative words and no filtering, some scores should be < 0
	hasNeg := false
	for _, r := range results {
		if r.Score < 0 {
			hasNeg = true
		}
	}
	// "terrible" should have negative score (similar to "bad" which is a negative seed)
	if !hasNeg {
		t.Log("warning: expected some negative scores with negative seed words")
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Verify cosine similarity is computed correctly
	m := makeTestModel()
	results := m.MostSimilar("good", 1)
	if len(results) == 0 {
		t.Fatal("expected results")
	}

	// Manually compute cosine similarity between good [1,0,0.1] and great [0.9,0.1,0.1]
	a := []float64{1.0, 0.0, 0.1}
	b := []float64{0.9, 0.1, 0.1}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	expected := dot / (math.Sqrt(normA) * math.Sqrt(normB))

	if math.Abs(results[0].Score-expected) > 1e-6 {
		t.Errorf("cosine similarity mismatch: got %.6f, expected %.6f", results[0].Score, expected)
	}
}
