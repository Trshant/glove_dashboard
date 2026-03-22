package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"

	"glove/internal/word2vec"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	os.Args = append(os.Args[:1], os.Args[2:]...)

	switch cmd {
	case "create":
		cmdCreate()
	case "add":
		cmdAdd()
	case "import":
		cmdImport()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: train <command> [options]

Commands:
  create    Create a new model from a corpus
  add       Add sentences to an existing model (incremental training)
  import    Import from word2vec text format (one-time migration)

Examples:
  train create --input corpus.txt --output case_data/IMDB --min-count 10
  train add --model case_data/IMDB --input new_sentences.txt
  train import --input case_data/IMDB/vectors.txt --output case_data/IMDB
`)
}

func cmdCreate() {
	fs := flag.NewFlagSet("create", flag.ExitOnError)
	input := fs.String("input", "", "Input corpus file (one sentence per line)")
	output := fs.String("output", "", "Output model directory")
	minCount := fs.Int("min-count", 10, "Minimum word frequency")
	vecSize := fs.Int("size", 100, "Vector dimensionality")
	window := fs.Int("window", 5, "Context window size")
	epochs := fs.Int("epochs", 5, "Training epochs")
	workers := fs.Int("workers", 4, "Number of parallel workers")
	fs.Parse(os.Args[1:])

	if *input == "" || *output == "" {
		fs.Usage()
		os.Exit(1)
	}

	sentences, err := readSentences(*input)
	if err != nil {
		log.Fatalf("Reading input: %v", err)
	}
	log.Printf("Read %d sentences", len(sentences))

	config := word2vec.TrainConfig{
		VectorSize: *vecSize,
		Window:     *window,
		MinCount:   *minCount,
		Workers:    *workers,
		Epochs:     *epochs,
		Alpha:      0.025,
		MinAlpha:   0.0001,
		NegSamples: 5,
	}

	model := word2vec.NewModel(config)
	log.Println("Training model...")
	model.CreateAndTrain(sentences)
	log.Printf("Vocabulary size: %d", model.Vocab.Size())

	if err := model.SaveModel(*output); err != nil {
		log.Fatalf("Saving model: %v", err)
	}
	log.Printf("Model saved to %s", *output)
}

func cmdAdd() {
	fs := flag.NewFlagSet("add", flag.ExitOnError)
	modelDir := fs.String("model", "", "Model directory")
	input := fs.String("input", "", "Input file with new sentences")
	fs.Parse(os.Args[1:])

	if *modelDir == "" || *input == "" {
		fs.Usage()
		os.Exit(1)
	}

	log.Printf("Loading model from %s...", *modelDir)
	model, err := word2vec.LoadModel(*modelDir)
	if err != nil {
		log.Fatalf("Loading model: %v", err)
	}
	log.Printf("Loaded: %d words", model.Vocab.Size())

	sentences, err := readSentences(*input)
	if err != nil {
		log.Fatalf("Reading input: %v", err)
	}
	log.Printf("Read %d new sentences", len(sentences))

	log.Println("Incremental training...")
	model.AddSentences(sentences)
	log.Printf("Vocabulary size: %d", model.Vocab.Size())

	if err := model.SaveModel(*modelDir); err != nil {
		log.Fatalf("Saving model: %v", err)
	}
	log.Printf("Model saved to %s", *modelDir)
}

func cmdImport() {
	fs := flag.NewFlagSet("import", flag.ExitOnError)
	input := fs.String("input", "", "Input word2vec text file (vectors.txt)")
	output := fs.String("output", "", "Output model directory")
	freqFile := fs.String("freq", "", "Optional vocab frequency JSON file")
	fs.Parse(os.Args[1:])

	if *input == "" || *output == "" {
		fs.Usage()
		os.Exit(1)
	}

	var model *word2vec.Model
	var err error

	if *freqFile != "" {
		model, err = word2vec.LoadWord2VecTextWithFreq(*input, *freqFile)
	} else {
		model, err = word2vec.LoadWord2VecText(*input)
	}
	if err != nil {
		log.Fatalf("Loading word2vec text: %v", err)
	}
	log.Printf("Imported: %d words, %d dimensions", model.Vocab.Size(), model.Config.VectorSize)

	if err := model.SaveModel(*output); err != nil {
		log.Fatalf("Saving model: %v", err)
	}
	log.Printf("Model saved to %s", *output)
}

// readSentences reads a file with one sentence per line, tokenizing each.
func readSentences(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var sentences [][]string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		tokens := word2vec.Tokenize(scanner.Text())
		if len(tokens) > 0 {
			sentences = append(sentences, tokens)
		}
	}
	return sentences, scanner.Err()
}
