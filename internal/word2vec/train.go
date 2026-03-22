package word2vec

import (
	"math"
	"math/rand"
	"sync"
)

// TrainConfig holds hyperparameters for Word2Vec CBOW training.
type TrainConfig struct {
	VectorSize int     `json:"vector_size"` // Embedding dimension (default: 100)
	Window     int     `json:"window"`      // Context window size (default: 5)
	MinCount   int     `json:"min_count"`   // Minimum word frequency (default: 10)
	Workers    int     `json:"workers"`     // Number of parallel workers (default: 4)
	Epochs     int     `json:"epochs"`      // Training epochs (default: 5)
	Alpha      float64 `json:"alpha"`       // Initial learning rate (default: 0.025)
	MinAlpha   float64 `json:"min_alpha"`   // Minimum learning rate (default: 0.0001)
	NegSamples int     `json:"neg_samples"` // Negative samples count (default: 5)
}

// DefaultConfig returns the default training configuration matching Gensim defaults.
func DefaultConfig() TrainConfig {
	return TrainConfig{
		VectorSize: 100,
		Window:     5,
		MinCount:   10,
		Workers:    4,
		Epochs:     5,
		Alpha:      0.025,
		MinAlpha:   0.0001,
		NegSamples: 5,
	}
}

// noiseTable holds the unigram distribution raised to the 0.75 power for negative sampling.
type noiseTable struct {
	table []int
}

// buildNoiseTable constructs a noise distribution table for negative sampling.
// Each word appears proportionally to freq^0.75 (matching Gensim/word2vec).
func buildNoiseTable(vocab *Vocab, tableSize int) *noiseTable {
	nt := &noiseTable{table: make([]int, tableSize)}

	// Compute total power
	var totalPow float64
	for _, w := range vocab.Words {
		totalPow += math.Pow(float64(vocab.Frequency[w]), 0.75)
	}

	// Fill table
	idx := 0
	cumulative := 0.0
	for i, w := range vocab.Words {
		cumulative += math.Pow(float64(vocab.Frequency[w]), 0.75) / totalPow
		bound := int(cumulative * float64(tableSize))
		if bound > tableSize {
			bound = tableSize
		}
		for idx < bound {
			nt.table[idx] = i
			idx++
		}
		_ = i // use i as the index for this word
	}
	// Fill remaining slots with last word
	for idx < tableSize {
		nt.table[idx] = len(vocab.Words) - 1
		idx++
	}

	return nt
}

func (nt *noiseTable) sample(rng *rand.Rand) int {
	return nt.table[rng.Intn(len(nt.table))]
}

// Train runs CBOW training with negative sampling on the given sentences.
func Train(model *Model, sentences [][]string) {
	vocabSize := model.Vocab.Size()
	if vocabSize == 0 {
		return
	}

	noise := buildNoiseTable(model.Vocab, 1e7)
	totalWords := 0
	for _, sent := range sentences {
		totalWords += len(sent)
	}

	totalTrainWords := float64(totalWords) * float64(model.Config.Epochs)

	var wordsProcessed int64
	var mu sync.Mutex

	// Split sentences across workers
	workers := model.Config.Workers
	if workers < 1 {
		workers = 1
	}

	chunkSize := (len(sentences) + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(sentences) {
			end = len(sentences)
		}
		if start >= end {
			continue
		}

		wg.Add(1)
		go func(chunk [][]string, seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			neu1 := make([]float32, model.Config.VectorSize)  // hidden layer (context average)
			neu1e := make([]float32, model.Config.VectorSize)  // error accumulator
			localProcessed := int64(0)

			for epoch := 0; epoch < model.Config.Epochs; epoch++ {
				for _, sent := range chunk {
					// Convert sentence to vocab indices, skipping unknown words
					var indices []int
					for _, w := range sent {
						if idx, ok := model.Vocab.Index[w]; ok {
							indices = append(indices, idx)
						}
					}
					if len(indices) < 2 {
						continue
					}

					for pos, targetIdx := range indices {
						// Compute learning rate
						mu.Lock()
						progress := float64(wordsProcessed+localProcessed) / totalTrainWords
						mu.Unlock()
						alpha := model.Config.Alpha - (model.Config.Alpha-model.Config.MinAlpha)*progress
						if alpha < model.Config.MinAlpha {
							alpha = model.Config.MinAlpha
						}

						// Dynamic window: random reduction like Gensim
						reducedWindow := rng.Intn(model.Config.Window)
						windowStart := pos - model.Config.Window + reducedWindow
						windowEnd := pos + model.Config.Window - reducedWindow + 1
						if windowStart < 0 {
							windowStart = 0
						}
						if windowEnd > len(indices) {
							windowEnd = len(indices)
						}

						// Compute context average (CBOW)
						for i := range neu1 {
							neu1[i] = 0
						}
						contextCount := 0
						for i := windowStart; i < windowEnd; i++ {
							if i == pos {
								continue
							}
							cIdx := indices[i]
							for j := 0; j < model.Config.VectorSize; j++ {
								neu1[j] += model.SynIn[cIdx][j]
							}
							contextCount++
						}
						if contextCount == 0 {
							continue
						}
						invCount := float32(1.0 / float64(contextCount))
						for j := range neu1 {
							neu1[j] *= invCount
						}

						// Clear error accumulator
						for i := range neu1e {
							neu1e[i] = 0
						}

						// Negative sampling: train on positive sample + negative samples
						for d := 0; d <= model.Config.NegSamples; d++ {
							var label float32
							var target int
							if d == 0 {
								// Positive sample
								target = targetIdx
								label = 1.0
							} else {
								// Negative sample
								target = noise.sample(rng)
								if target == targetIdx {
									continue
								}
								label = 0.0
							}

							// Dot product of hidden layer and output vector
							var dot float32
							for j := 0; j < model.Config.VectorSize; j++ {
								dot += neu1[j] * model.SynOut[target][j]
							}

							// Sigmoid
							var g float32
							if dot > 6 {
								g = (label - 1.0) * float32(alpha)
							} else if dot < -6 {
								g = (label - 0.0) * float32(alpha)
							} else {
								expVal := float32(math.Exp(float64(dot)))
								g = (label - expVal/(expVal+1.0)) * float32(alpha)
							}

							// Accumulate error for input vectors
							for j := 0; j < model.Config.VectorSize; j++ {
								neu1e[j] += g * model.SynOut[target][j]
							}
							// Update output vector
							for j := 0; j < model.Config.VectorSize; j++ {
								model.SynOut[target][j] += g * neu1[j]
							}
						}

						// Update input vectors for context words
						for i := windowStart; i < windowEnd; i++ {
							if i == pos {
								continue
							}
							cIdx := indices[i]
							for j := 0; j < model.Config.VectorSize; j++ {
								model.SynIn[cIdx][j] += neu1e[j]
							}
						}

						localProcessed++
					}
				}
			}

			mu.Lock()
			wordsProcessed += localProcessed
			mu.Unlock()
		}(sentences[start:end], int64(w*17+42))
	}

	wg.Wait()
}
