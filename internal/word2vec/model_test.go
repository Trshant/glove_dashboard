package word2vec

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTrainAndSaveLoad(t *testing.T) {
	sentences := [][]string{
		{"the", "cat", "sat", "on", "the", "mat"},
		{"the", "dog", "sat", "on", "the", "log"},
		{"the", "cat", "and", "the", "dog", "are", "friends"},
		{"the", "mat", "is", "on", "the", "floor"},
		{"a", "cat", "sat", "on", "a", "mat"},
		{"the", "dog", "and", "cat", "played"},
		{"on", "the", "mat", "sat", "the", "cat"},
		{"the", "dog", "sat", "on", "the", "mat"},
	}

	config := TrainConfig{
		VectorSize: 10,
		Window:     2,
		MinCount:   2,
		Workers:    2,
		Epochs:     3,
		Alpha:      0.025,
		MinAlpha:   0.0001,
		NegSamples: 3,
	}

	model := NewModel(config)
	model.CreateAndTrain(sentences)

	if model.Vocab.Size() == 0 {
		t.Fatal("expected non-empty vocab after training")
	}

	// Test save/load round-trip
	dir := t.TempDir()
	if err := model.SaveModel(dir); err != nil {
		t.Fatalf("saving model: %v", err)
	}

	// Verify files exist
	for _, f := range []string{"model.bin", "vectors.txt", "vocab_freq.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("expected %s to exist: %v", f, err)
		}
	}

	// Load from binary
	loaded, err := LoadModel(dir)
	if err != nil {
		t.Fatalf("loading model: %v", err)
	}

	if loaded.Vocab.Size() != model.Vocab.Size() {
		t.Errorf("vocab size mismatch: got %d, want %d", loaded.Vocab.Size(), model.Vocab.Size())
	}
	if loaded.Config.VectorSize != model.Config.VectorSize {
		t.Errorf("vector size mismatch: got %d, want %d", loaded.Config.VectorSize, model.Config.VectorSize)
	}

	// Verify vectors are preserved
	for i, word := range model.Vocab.Words {
		loadedIdx, ok := loaded.Vocab.Index[word]
		if !ok {
			t.Errorf("word %q missing from loaded model", word)
			continue
		}
		for j := 0; j < config.VectorSize; j++ {
			if model.SynIn[i][j] != loaded.SynIn[loadedIdx][j] {
				t.Errorf("vector mismatch for %q[%d]: got %f, want %f", word, j, loaded.SynIn[loadedIdx][j], model.SynIn[i][j])
			}
		}
	}
}

func TestLoadWord2VecText(t *testing.T) {
	// Create a model, save as text, reload
	sentences := [][]string{
		{"the", "cat", "sat", "on", "the", "mat"},
		{"the", "dog", "sat", "on", "the", "log"},
		{"a", "cat", "sat", "on", "a", "mat"},
		{"the", "dog", "sat", "on", "the", "mat"},
	}

	config := TrainConfig{
		VectorSize: 5,
		Window:     2,
		MinCount:   2,
		Workers:    1,
		Epochs:     2,
		Alpha:      0.025,
		MinAlpha:   0.0001,
		NegSamples: 2,
	}

	model := NewModel(config)
	model.CreateAndTrain(sentences)

	dir := t.TempDir()
	vecPath := filepath.Join(dir, "vectors.txt")
	if err := model.SaveWord2VecText(vecPath); err != nil {
		t.Fatalf("saving word2vec text: %v", err)
	}

	loaded, err := LoadWord2VecText(vecPath)
	if err != nil {
		t.Fatalf("loading word2vec text: %v", err)
	}

	if loaded.Vocab.Size() != model.Vocab.Size() {
		t.Errorf("vocab size mismatch: got %d, want %d", loaded.Vocab.Size(), model.Vocab.Size())
	}
}

func TestIncrementalTraining(t *testing.T) {
	sentences := [][]string{
		{"the", "cat", "sat", "on", "the", "mat"},
		{"the", "dog", "sat", "on", "the", "log"},
		{"a", "cat", "sat", "on", "a", "mat"},
		{"the", "dog", "sat", "on", "the", "mat"},
	}

	config := TrainConfig{
		VectorSize: 10,
		Window:     2,
		MinCount:   2,
		Workers:    1,
		Epochs:     2,
		Alpha:      0.025,
		MinAlpha:   0.0001,
		NegSamples: 2,
	}

	model := NewModel(config)
	model.CreateAndTrain(sentences)
	origSize := model.Vocab.Size()

	// Add new sentences with new words
	newSentences := [][]string{
		{"the", "bird", "flew", "over", "the", "mat"},
		{"the", "bird", "and", "the", "cat"},
		{"a", "bird", "sat", "on", "the", "log"},
	}

	model.AddSentences(newSentences)

	// "bird" appeared 3 times (>= minCount=2), should be in vocab now
	if !model.Vocab.Contains("bird") {
		t.Error("expected 'bird' in vocab after incremental training")
	}

	if model.Vocab.Size() <= origSize {
		t.Error("expected vocab to grow after incremental training")
	}

	// Model should still work for similarity queries
	results := model.MostSimilar("the", 3)
	if len(results) == 0 {
		t.Error("expected similarity results after incremental training")
	}
}
