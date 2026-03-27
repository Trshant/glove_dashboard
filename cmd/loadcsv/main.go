package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"glove/internal/elastic"
)

func main() {
	csvPath := flag.String("csv", "", "Path to CSV file (must have a 'review' column)")
	esAddr := flag.String("es-addr", envOrDefault("ES_ADDR", "http://localhost:9200"), "Elasticsearch address")
	esUser := flag.String("es-user", envOrDefault("ES_USER", "elastic"), "Elasticsearch username")
	esPass := flag.String("es-pass", envOrDefault("ES_PASS", "JqdGMYPXDGbkFIMLX"), "Elasticsearch password")
	esIndex := flag.String("es-index", envOrDefault("ES_INDEX", "imdb"), "Elasticsearch index name")
	batchSize := flag.Int("batch", 500, "Bulk insert batch size")
	flag.Parse()

	if *csvPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: loadcsv --csv <file.csv> [--es-index imdb]")
		os.Exit(1)
	}

	// Read CSV
	log.Printf("Reading %s...", *csvPath)
	docs, err := readCSV(*csvPath)
	if err != nil {
		log.Fatalf("Reading CSV: %v", err)
	}
	log.Printf("Read %d reviews", len(docs))

	// Connect to ES
	log.Printf("Connecting to Elasticsearch at %s...", *esAddr)
	es, err := elastic.NewClient(*esAddr, *esUser, *esPass, *esIndex)
	if err != nil {
		log.Fatalf("ES connection: %v", err)
	}

	// Check existing doc count
	count, _ := es.DocCount()
	if count > 0 {
		log.Printf("Index %q already has %d documents. Loading %d more.", *esIndex, count, len(docs))
	}

	// Bulk insert
	log.Println("Loading documents into Elasticsearch...")
	err = es.BulkInsert(docs, *batchSize, func(n int) {
		log.Printf("  indexed %d / %d", n, len(docs))
	})
	if err != nil {
		log.Fatalf("Bulk insert: %v", err)
	}

	finalCount, _ := es.DocCount()
	log.Printf("Done. Index %q now has %d documents.", *esIndex, finalCount)
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// readCSV reads a CSV and returns the "review" column contents, with HTML tags stripped.
func readCSV(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.LazyQuotes = true

	// Read header to find "review" column index
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	reviewIdx := -1
	for i, col := range header {
		if strings.TrimSpace(strings.ToLower(col)) == "review" {
			reviewIdx = i
			break
		}
	}
	if reviewIdx < 0 {
		return nil, fmt.Errorf("no 'review' column found in CSV header: %v", header)
	}

	var docs []string
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		if reviewIdx < len(record) {
			text := cleanHTML(record[reviewIdx])
			text = strings.TrimSpace(text)
			if text != "" {
				docs = append(docs, text)
			}
		}
	}

	return docs, nil
}

// cleanHTML strips <br /> and other basic HTML tags from review text.
func cleanHTML(s string) string {
	s = strings.ReplaceAll(s, "<br />", " ")
	s = strings.ReplaceAll(s, "<br/>", " ")
	s = strings.ReplaceAll(s, "<br>", " ")
	// Strip any remaining HTML tags
	var out strings.Builder
	inTag := false
	for _, c := range s {
		if c == '<' {
			inTag = true
			continue
		}
		if c == '>' {
			inTag = false
			continue
		}
		if !inTag {
			out.WriteRune(c)
		}
	}
	return out.String()
}
