package word2vec

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Model holds the Word2Vec model: vocabulary, weight matrices, and config.
type Model struct {
	Vocab  *Vocab
	SynIn  [][]float32 // Input weight matrix (vocab_size × vector_size)
	SynOut [][]float32 // Output weight matrix (vocab_size × vector_size)
	Config TrainConfig
}

// NewModel creates a new untrained model with the given config.
func NewModel(config TrainConfig) *Model {
	return &Model{
		Config: config,
	}
}

// InitWeights initializes weight matrices with small random values.
func (m *Model) InitWeights() {
	vocabSize := m.Vocab.Size()
	vecSize := m.Config.VectorSize

	m.SynIn = make([][]float32, vocabSize)
	m.SynOut = make([][]float32, vocabSize)

	rng := rand.New(rand.NewSource(1))
	for i := 0; i < vocabSize; i++ {
		m.SynIn[i] = make([]float32, vecSize)
		m.SynOut[i] = make([]float32, vecSize)
		for j := 0; j < vecSize; j++ {
			m.SynIn[i][j] = (rng.Float32() - 0.5) / float32(vecSize)
		}
		// SynOut initialized to zero (default)
	}
}

// ExtendWeights extends weight matrices for new vocabulary entries added via ExtendVocab.
// Existing rows are preserved; new rows are initialized with small random values.
func (m *Model) ExtendWeights(oldSize int) {
	newSize := m.Vocab.Size()
	vecSize := m.Config.VectorSize
	rng := rand.New(rand.NewSource(int64(newSize)))

	for i := oldSize; i < newSize; i++ {
		row := make([]float32, vecSize)
		for j := range row {
			row[j] = (rng.Float32() - 0.5) / float32(vecSize)
		}
		m.SynIn = append(m.SynIn, row)
		m.SynOut = append(m.SynOut, make([]float32, vecSize))
	}
}

// CreateAndTrain builds vocabulary from sentences and trains the model.
func (m *Model) CreateAndTrain(sentences [][]string) {
	m.Vocab = BuildVocab(sentences, m.Config.MinCount)
	m.InitWeights()
	Train(m, sentences)
}

// AddSentences extends the vocabulary with new sentences and trains on them (incremental training).
func (m *Model) AddSentences(sentences [][]string) {
	oldSize := m.Vocab.Size()
	m.Vocab.ExtendVocab(sentences)
	m.ExtendWeights(oldSize)
	Train(m, sentences)
}

// SaveModel saves the model in a Go-native binary format plus a portable word2vec text file.
func (m *Model) SaveModel(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Save binary format
	if err := m.saveBinary(filepath.Join(dir, "model.bin")); err != nil {
		return fmt.Errorf("saving binary: %w", err)
	}

	// Save word2vec text format for portability
	if err := m.SaveWord2VecText(filepath.Join(dir, "vectors.txt")); err != nil {
		return fmt.Errorf("saving vectors.txt: %w", err)
	}

	// Save vocab frequencies
	if err := m.saveVocabFreq(filepath.Join(dir, "vocab_freq.json")); err != nil {
		return fmt.Errorf("saving vocab_freq: %w", err)
	}

	return nil
}

// LoadModel loads a model from binary format.
func LoadModel(dir string) (*Model, error) {
	return loadBinary(filepath.Join(dir, "model.bin"))
}

// LoadWord2VecText imports vectors from standard word2vec text format.
// This is used for one-time migration from Python-exported models.
func LoadWord2VecText(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB line buffer

	// First line: vocabSize vectorSize
	if !scanner.Scan() {
		return nil, fmt.Errorf("empty file")
	}
	header := strings.Fields(scanner.Text())
	if len(header) != 2 {
		return nil, fmt.Errorf("invalid header: %q", scanner.Text())
	}
	vocabSize, err := strconv.Atoi(header[0])
	if err != nil {
		return nil, fmt.Errorf("invalid vocab size: %w", err)
	}
	vecSize, err := strconv.Atoi(header[1])
	if err != nil {
		return nil, fmt.Errorf("invalid vector size: %w", err)
	}

	config := DefaultConfig()
	config.VectorSize = vecSize

	vocab := NewVocab(config.MinCount)
	synIn := make([][]float32, 0, vocabSize)

	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != vecSize+1 {
			continue // skip malformed lines
		}
		word := fields[0]
		vec := make([]float32, vecSize)
		for i := 0; i < vecSize; i++ {
			v, err := strconv.ParseFloat(fields[i+1], 32)
			if err != nil {
				return nil, fmt.Errorf("parsing vector for %q: %w", word, err)
			}
			vec[i] = float32(v)
		}

		vocab.Index[word] = len(vocab.Words)
		vocab.Words = append(vocab.Words, word)
		vocab.Frequency[word] = 1 // Default frequency; will be overridden by vocab_freq.json if available
		synIn = append(synIn, vec)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	m := &Model{
		Vocab:  vocab,
		SynIn:  synIn,
		SynOut: make([][]float32, len(synIn)), // Initialize empty output vectors
		Config: config,
	}

	// Initialize output vectors to zero
	for i := range m.SynOut {
		m.SynOut[i] = make([]float32, vecSize)
	}

	return m, nil
}

// LoadWord2VecTextWithFreq loads vectors from text format and vocab frequencies from JSON.
func LoadWord2VecTextWithFreq(vectorsPath, freqPath string) (*Model, error) {
	m, err := LoadWord2VecText(vectorsPath)
	if err != nil {
		return nil, err
	}

	// Load frequency file if it exists
	data, err := os.ReadFile(freqPath)
	if err != nil {
		if os.IsNotExist(err) {
			return m, nil // No freq file, use defaults
		}
		return nil, err
	}

	var freq map[string]int
	if err := json.Unmarshal(data, &freq); err != nil {
		return nil, fmt.Errorf("parsing vocab_freq.json: %w", err)
	}

	for word, count := range freq {
		if _, ok := m.Vocab.Index[word]; ok {
			m.Vocab.Frequency[word] = count
		}
	}

	return m, nil
}

// SaveWord2VecText saves vectors in standard word2vec text format.
func (m *Model) SaveWord2VecText(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	fmt.Fprintf(w, "%d %d\n", m.Vocab.Size(), m.Config.VectorSize)

	for i, word := range m.Vocab.Words {
		fmt.Fprintf(w, "%s", word)
		for j := 0; j < m.Config.VectorSize; j++ {
			fmt.Fprintf(w, " %g", m.SynIn[i][j])
		}
		fmt.Fprintln(w)
	}

	return w.Flush()
}

func (m *Model) saveVocabFreq(path string) error {
	data, err := json.Marshal(m.Vocab.Frequency)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Binary format: config JSON + vocab + SynIn + SynOut
func (m *Model) saveBinary(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)

	// Write config as JSON
	configData, err := json.Marshal(m.Config)
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(len(configData))); err != nil {
		return err
	}
	if _, err := w.Write(configData); err != nil {
		return err
	}

	// Write vocab size
	vocabSize := int32(m.Vocab.Size())
	if err := binary.Write(w, binary.LittleEndian, vocabSize); err != nil {
		return err
	}

	// Write words and frequencies
	for _, word := range m.Vocab.Words {
		wordBytes := []byte(word)
		if err := binary.Write(w, binary.LittleEndian, int32(len(wordBytes))); err != nil {
			return err
		}
		if _, err := w.Write(wordBytes); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, int32(m.Vocab.Frequency[word])); err != nil {
			return err
		}
	}

	// Write SynIn matrix
	for i := 0; i < int(vocabSize); i++ {
		if err := binary.Write(w, binary.LittleEndian, m.SynIn[i]); err != nil {
			return err
		}
	}

	// Write SynOut matrix
	for i := 0; i < int(vocabSize); i++ {
		if err := binary.Write(w, binary.LittleEndian, m.SynOut[i]); err != nil {
			return err
		}
	}

	return w.Flush()
}

func loadBinary(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)

	// Read config
	var configLen int32
	if err := binary.Read(r, binary.LittleEndian, &configLen); err != nil {
		return nil, err
	}
	configData := make([]byte, configLen)
	if _, err := io.ReadFull(r, configData); err != nil {
		return nil, err
	}
	var config TrainConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, err
	}

	// Read vocab size
	var vocabSize int32
	if err := binary.Read(r, binary.LittleEndian, &vocabSize); err != nil {
		return nil, err
	}

	vocab := NewVocab(config.MinCount)
	for i := 0; i < int(vocabSize); i++ {
		var wordLen int32
		if err := binary.Read(r, binary.LittleEndian, &wordLen); err != nil {
			return nil, err
		}
		wordBytes := make([]byte, wordLen)
		if _, err := io.ReadFull(r, wordBytes); err != nil {
			return nil, err
		}
		var freq int32
		if err := binary.Read(r, binary.LittleEndian, &freq); err != nil {
			return nil, err
		}
		word := string(wordBytes)
		vocab.Index[word] = len(vocab.Words)
		vocab.Words = append(vocab.Words, word)
		vocab.Frequency[word] = int(freq)
	}

	// Read SynIn
	synIn := make([][]float32, vocabSize)
	for i := 0; i < int(vocabSize); i++ {
		synIn[i] = make([]float32, config.VectorSize)
		if err := binary.Read(r, binary.LittleEndian, synIn[i]); err != nil {
			return nil, err
		}
	}

	// Read SynOut
	synOut := make([][]float32, vocabSize)
	for i := 0; i < int(vocabSize); i++ {
		synOut[i] = make([]float32, config.VectorSize)
		if err := binary.Read(r, binary.LittleEndian, synOut[i]); err != nil {
			return nil, err
		}
	}

	return &Model{
		Vocab:  vocab,
		SynIn:  synIn,
		SynOut: synOut,
		Config: config,
	}, nil
}

// NormalizedVectors returns L2-normalized copies of SynIn vectors for cosine similarity.
func (m *Model) NormalizedVectors() [][]float32 {
	norms := make([][]float32, len(m.SynIn))
	for i, vec := range m.SynIn {
		norms[i] = make([]float32, len(vec))
		var norm float64
		for _, v := range vec {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			copy(norms[i], vec)
			continue
		}
		invNorm := float32(1.0 / norm)
		for j, v := range vec {
			norms[i][j] = v * invNorm
		}
	}
	return norms
}
