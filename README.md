# Glove — Semantic Search with Word2Vec + Elasticsearch

A web application that combines Word2Vec word embeddings with Elasticsearch full-text search. Enter a term, explore semantically similar words, mark them as positive or negative, and the app builds an Elasticsearch query to search an IMDB review corpus.

## Prerequisites

- **Go 1.22+**
- **Docker** (for Elasticsearch)

## Quick Start

```bash
# 1. Start Elasticsearch
docker compose up -d elasticsearch

# 2. Wait for ES to be healthy (~30 seconds)
docker compose ps   # should show "healthy"

# 3. Run the server
go run cmd/server/main.go

# 4. Open http://localhost:8001
```

## Setup in Detail

### 1. Start Elasticsearch

The project uses Elasticsearch 8.12 with security enabled. Docker Compose handles everything:

```bash
docker compose up -d elasticsearch
```

Default credentials (configured in `docker-compose.yml`):
- **Username**: `elastic`
- **Password**: `JqdGMYPXDGbkFIMLX`
- **Address**: `http://localhost:9200`

### 2. Prepare a Model

The server needs a trained Word2Vec model in a model directory. You have three options:

#### Option A: Use the pre-trained IMDB model (included)

The repo ships with a pre-trained model at `case_data/IMDB/`. Just start the server — it will load automatically.

#### Option B: Train a new model from a corpus

Prepare a text file with one sentence per line, then:

```bash
go run cmd/train/main.go create \
  --input corpus.txt \
  --output case_data/MyModel \
  --min-count 10 \
  --size 100 \
  --window 5 \
  --epochs 5 \
  --workers 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Path to corpus file, one sentence per line |
| `--output` | (required) | Directory to save the model |
| `--min-count` | 10 | Ignore words appearing fewer than this many times |
| `--size` | 100 | Dimensionality of word vectors |
| `--window` | 5 | Context window size (words on each side) |
| `--epochs` | 5 | Number of training passes over the corpus |
| `--workers` | 4 | Parallel training threads |

#### Option C: Import an existing word2vec text file

If you have vectors in the standard word2vec text format (first line: `vocab_size dimensions`, then `word float float ...` per line):

```bash
go run cmd/train/main.go import \
  --input vectors.txt \
  --output case_data/MyModel
```

If you also have a vocab frequency JSON file (`{"word": count, ...}`), pass it with `--freq` to enable accurate incremental training later:

```bash
go run cmd/train/main.go import \
  --input vectors.txt \
  --freq vocab_freq.json \
  --output case_data/MyModel
```

### 3. Run the Server

```bash
go run cmd/server/main.go
```

The server accepts these flags (all have sensible defaults):

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--port` | `PORT` | `8001` | HTTP port |
| `--model` | `MODEL_DIR` | `case_data/IMDB` | Model directory path |
| `--es-addr` | `ES_ADDR` | `http://localhost:9200` | Elasticsearch URL |
| `--es-user` | `ES_USER` | `elastic` | ES username |
| `--es-pass` | `ES_PASS` | `JqdGMYPXDGbkFIMLX` | ES password |
| `--es-index` | `ES_INDEX` | `imdb` | ES index name |

To use a different model:

```bash
go run cmd/server/main.go --model case_data/MyModel --es-index myindex
```

Environment variables work too (useful for Docker):

```bash
MODEL_DIR=case_data/MyModel ES_INDEX=myindex go run cmd/server/main.go
```

## Adding New Data to an Existing Model (Incremental Training)

One of the key features is the ability to add new sentences to an already-trained model without retraining from scratch. New words are added to the vocabulary, weight matrices are extended, and training runs on just the new sentences:

```bash
go run cmd/train/main.go add \
  --model case_data/IMDB \
  --input new_reviews.txt
```

The input file should have one sentence per line. The model is updated in place.

This is useful when your corpus grows over time — for example, indexing new reviews each week. Existing word vectors are refined and new words are learned without a full retrain.

## Running with Docker Compose (Full Stack)

To run both the app and Elasticsearch together:

```bash
docker compose up
```

This builds the Go app from the `Dockerfile`, starts ES, waits for it to be healthy, then starts the server on port 8001.

## Building Binaries

```bash
make build
```

Produces `bin/server` and `bin/train`.

## Running Tests

```bash
go test ./...
```

## Project Structure

```
cmd/
  server/main.go         Web server entry point
  train/main.go          CLI for training and importing models
internal/
  word2vec/              Custom Word2Vec (CBOW + negative sampling)
    model.go             Model struct, save/load (binary + text formats)
    train.go             Training engine with parallel workers
    vocab.go             Vocabulary management with incremental extension
    inference.go         Cosine similarity search and aggregation
    tokenizer.go         \w+ lowercase tokenizer
  elastic/client.go      Elasticsearch client (index, search)
  query/builder.go       Bool query builder from positive/negative terms
  handlers/routes.go     HTTP route handlers
templates/               HTML templates (Go html/template + HTMX)
case_data/               Model data directories
```

## How It Works

1. You type a word (e.g. "good") into the search box
2. The server finds the 15 most semantically similar words using cosine similarity over Word2Vec embeddings
3. You check words as **positive** (include in search) or **negative** (exclude)
4. The app builds an Elasticsearch bool query combining your selections and searches the IMDB corpus
5. Matching reviews are displayed

The frontend uses HTMX for interactivity — no JavaScript framework needed. Checkbox changes trigger a search after a 1-second debounce.

## Model File Formats

Each model directory contains:

| File | Description |
|------|-------------|
| `model.bin` | Binary format (fast loading) — used by default |
| `vectors.txt` | Standard word2vec text format (portable, human-readable) — fallback |
| `vocab_freq.json` | Word frequency counts (needed for accurate incremental training) |

The server loads `model.bin` if present, otherwise falls back to `vectors.txt`.
