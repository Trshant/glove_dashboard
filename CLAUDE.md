# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A semantic search application that combines Word2Vec word embeddings with Elasticsearch full-text search. Users enter a term, get semantically similar words, mark them as positive/negative, and the app builds an Elasticsearch bool query to search an IMDB review corpus.

## Commands

```bash
# Build
make build

# Run the web server (starts on http://localhost:8001)
make run
# Or directly:
go run cmd/server/main.go

# Run tests
go test ./...

# Training CLI
go run cmd/train/main.go create --input corpus.txt --output case_data/IMDB --min-count 10
go run cmd/train/main.go add --model case_data/IMDB --input new_sentences.txt
go run cmd/train/main.go import --input vectors.txt --output case_data/IMDB

# Docker (includes Elasticsearch)
docker-compose up
```

## Prerequisites

- Go 1.22+
- Elasticsearch running at localhost:9200 (with basic auth credentials, see docker-compose.yml)
- Pre-trained Word2Vec model in `case_data/IMDB/` (model.bin or vectors.txt + vocab_freq.json)

## Architecture

**Stack**: Go (chi router) + html/template + HTMX (frontend interactivity) + PicoCSS (styling)

**Data flow**:
1. User submits a term → `POST /submit-form` → `Model.GetSimilarWords()` returns similar words via Word2Vec cosine similarity
2. User marks words as positive/negative → `POST /submit-search` → `query.BuildQuery()` builds an ES bool query → `elastic.Client.Search()` returns matching documents

**Key packages**:
- `cmd/server/main.go` — Web server entry point
- `cmd/train/main.go` — CLI for training / importing models
- `internal/word2vec/` — Custom Word2Vec implementation (CBOW + negative sampling) with incremental training support
  - `model.go` — Model struct, save/load (binary + word2vec text format)
  - `train.go` — CBOW training with negative sampling, parallel workers
  - `vocab.go` — Vocabulary: build, extend (incremental)
  - `inference.go` — MostSimilar, GetSimilarWords (cosine similarity + aggregation)
  - `tokenizer.go` — \w+ lowercase tokenizer
- `internal/elastic/client.go` — ES 8.x client: index management, search (BM25 similarity)
- `internal/query/builder.go` — Elasticsearch bool query builder from positive/negative term lists
- `internal/handlers/routes.go` — HTTP handlers for 3 routes (GET /, POST /submit-form, POST /submit-search)
- `templates/` — Go html/template + HTMX templates
