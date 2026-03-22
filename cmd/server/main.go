package main

import (
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"glove/internal/elastic"
	"glove/internal/handlers"
	"glove/internal/word2vec"
)

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	port := flag.String("port", envOrDefault("PORT", "8001"), "HTTP server port")
	modelDir := flag.String("model", envOrDefault("MODEL_DIR", "case_data/IMDB"), "Path to model directory")
	esAddr := flag.String("es-addr", envOrDefault("ES_ADDR", "http://localhost:9200"), "Elasticsearch address")
	esUser := flag.String("es-user", envOrDefault("ES_USER", "elastic"), "Elasticsearch username")
	esPass := flag.String("es-pass", envOrDefault("ES_PASS", "JqdGMYPXDGbkFIMLX"), "Elasticsearch password")
	esIndex := flag.String("es-index", envOrDefault("ES_INDEX", "imdb"), "Elasticsearch index name")
	flag.Parse()

	// Load model
	log.Printf("Loading model from %s...", *modelDir)
	var model *word2vec.Model
	var err error

	binPath := filepath.Join(*modelDir, "model.bin")
	vecPath := filepath.Join(*modelDir, "vectors.txt")
	freqPath := filepath.Join(*modelDir, "vocab_freq.json")

	if _, err := os.Stat(binPath); err == nil {
		// Prefer binary format (faster)
		model, err = word2vec.LoadModel(*modelDir)
		if err != nil {
			log.Fatalf("Failed to load binary model: %v", err)
		}
	} else if _, err := os.Stat(vecPath); err == nil {
		// Fall back to word2vec text format
		model, err = word2vec.LoadWord2VecTextWithFreq(vecPath, freqPath)
		if err != nil {
			log.Fatalf("Failed to load text model: %v", err)
		}
		log.Println("Loaded from word2vec text format")
	} else {
		log.Fatalf("No model found in %s (expected model.bin or vectors.txt)", *modelDir)
	}

	log.Printf("Model loaded: %d words, %d dimensions", model.Vocab.Size(), model.Config.VectorSize)

	// Connect to Elasticsearch
	log.Printf("Connecting to Elasticsearch at %s...", *esAddr)
	es, err := elastic.NewClient(*esAddr, *esUser, *esPass, *esIndex)
	if err != nil {
		log.Fatalf("Failed to connect to Elasticsearch: %v", err)
	}
	log.Println("Elasticsearch connected")

	// Parse templates
	tmpl := template.Must(template.ParseGlob("templates/*.html"))

	// Start server
	r := handlers.NewRouter(model, es, tmpl)
	addr := fmt.Sprintf(":%s", *port)
	log.Printf("Server starting on %s", addr)
	if err := http.ListenAndServe(addr, r); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
