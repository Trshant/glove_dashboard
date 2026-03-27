# Glove — Semantic Search with Word2Vec + Elasticsearch

A web application that combines Word2Vec word embeddings with Elasticsearch full-text search. Enter a term, explore semantically similar words, mark them as positive or negative, and the app builds an Elasticsearch query to search an IMDB review corpus.

## Prerequisites

- **Go 1.22+**
- **Docker** (for Elasticsearch)

## Setup from Scratch

Follow these steps in order to get a working system from a fresh clone.

### Step 1: Start Elasticsearch

```bash
docker compose up -d elasticsearch
```

Wait for it to be healthy (~30 seconds):

```bash
docker compose ps   # status should show "healthy"
```

Default credentials (configured in `docker-compose.yml`):
- **Username**: `elastic`
- **Password**: `JqdGMYPXDGbkFIMLX`
- **Address**: `http://localhost:9200`

### Step 2: Load your data into Elasticsearch

The search corpus needs to be indexed into Elasticsearch before the app can return results. The `loadcsv` tool reads a CSV file with a `review` column and bulk-inserts every row into ES:

```bash
go run cmd/loadcsv/main.go --csv "case_data/IMDB/IMDB Dataset.csv"
```

This loads all 50,000 IMDB reviews into the `imdb` index. It takes about 6 seconds.

To load into a different index:

```bash
go run cmd/loadcsv/main.go --csv my_data.csv --es-index myindex
```

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--csv` | | (required) | Path to CSV file. Must have a `review` column |
| `--es-index` | `ES_INDEX` | `imdb` | Elasticsearch index name |
| `--es-addr` | `ES_ADDR` | `http://localhost:9200` | Elasticsearch URL |
| `--es-user` | `ES_USER` | `elastic` | ES username |
| `--es-pass` | `ES_PASS` | `JqdGMYPXDGbkFIMLX` | ES password |
| `--batch` | | `500` | Bulk insert batch size |

### Step 3: Train the Word2Vec model

The server needs word vectors to find semantically similar words. The easiest way is to train directly from the data you just loaded into Elasticsearch:

```bash
go run cmd/train/main.go from-es \
  --es-index imdb \
  --output case_data/IMDB \
  --min-count 10
```

This scrolls through all documents in the index, tokenizes them, and trains a model. No intermediate file needed.

Alternatively, you can train from a local text file (one sentence per line):

```bash
go run cmd/train/main.go create \
  --input corpus.txt \
  --output case_data/IMDB \
  --min-count 10
```

Both commands accept the same training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | (required) | Directory to save the model |
| `--min-count` | 10 | Ignore words appearing fewer than this many times |
| `--size` | 100 | Dimensionality of word vectors |
| `--window` | 5 | Context window size (words on each side) |
| `--epochs` | 5 | Number of training passes over the corpus |
| `--workers` | 4 | Parallel training threads |

The `from-es` command also accepts `--es-addr`, `--es-user`, `--es-pass`, and `--es-index` (same defaults as the server).

If you already have vectors in standard word2vec text format, you can import them instead:

```bash
go run cmd/train/main.go import \
  --input vectors.txt \
  --freq vocab_freq.json \
  --output case_data/IMDB
```

The repo ships with a pre-trained IMDB model in `case_data/IMDB/` — if you just want to try the app, skip this step.

### Step 4: Start the server

```bash
go run cmd/server/main.go
```

Open **http://localhost:8001** in your browser.

Type a word (e.g. "good"), click **Analyze**, check some words as positive or negative, and search results will appear from the IMDB corpus.

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--port` | `PORT` | `8001` | HTTP port |
| `--model` | `MODEL_DIR` | `case_data/IMDB` | Model directory path |
| `--es-addr` | `ES_ADDR` | `http://localhost:9200` | Elasticsearch URL |
| `--es-user` | `ES_USER` | `elastic` | ES username |
| `--es-pass` | `ES_PASS` | `JqdGMYPXDGbkFIMLX` | ES password |
| `--es-index` | `ES_INDEX` | `imdb` | ES index name |

---

## Adding a New Dataset

You can use this system with any text corpus, not just IMDB. Here's how to set up a second dataset alongside the existing one.

### 1. Prepare your CSV

Your CSV needs a `review` column (the text content to be searched). Place it somewhere convenient:

```
case_data/MyCorpus/data.csv
```

### 2. Load it into a new ES index

```bash
go run cmd/loadcsv/main.go --csv case_data/MyCorpus/data.csv --es-index mycorpus
```

### 3. Train vectors from the data you just loaded

```bash
go run cmd/train/main.go from-es \
  --es-index mycorpus \
  --output case_data/MyCorpus \
  --min-count 10
```

### 4. Run the server pointing at the new dataset

```bash
go run cmd/server/main.go --model case_data/MyCorpus --es-index mycorpus
```

---

## Incremental Training

You can add new sentences to an already-trained model without retraining from scratch. New words are added to the vocabulary, weight matrices are extended, and training runs on just the new sentences:

```bash
go run cmd/train/main.go add \
  --model case_data/IMDB \
  --input new_reviews.txt
```

The input file should have one sentence per line. The model is updated in place.

This is useful when your corpus grows over time — for example, indexing new reviews each week. Existing word vectors are refined and new words are learned without a full retrain.

Don't forget to also load the new text into Elasticsearch so it appears in search results:

```bash
go run cmd/loadcsv/main.go --csv new_reviews.csv
```

---

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
  loadcsv/main.go        CLI for loading CSV data into Elasticsearch
internal/
  word2vec/              Custom Word2Vec (CBOW + negative sampling)
    model.go             Model struct, save/load (binary + text formats)
    train.go             Training engine with parallel workers
    vocab.go             Vocabulary management with incremental extension
    inference.go         Cosine similarity search and aggregation
    tokenizer.go         \w+ lowercase tokenizer
  elastic/client.go      Elasticsearch client (index, bulk insert, search)
  query/builder.go       Bool query builder from positive/negative terms
  handlers/routes.go     HTTP route handlers
templates/               HTML templates (Go html/template + HTMX)
case_data/               Model data and corpus directories
```

## How It Works

1. You type a word (e.g. "good") into the search box and click **Analyze**
2. The server finds the 15 most semantically similar words using cosine similarity over Word2Vec embeddings
3. You check words as **positive** (include in search) or **negative** (exclude)
4. The app builds an Elasticsearch bool query combining your selections and searches the corpus
5. Matching documents are displayed

The frontend uses HTMX for interactivity — no JavaScript framework needed. Checkbox changes trigger a search after a 1-second debounce.

## Model File Formats

Each model directory contains:

| File | Description |
|------|-------------|
| `model.bin` | Binary format (fast loading) — used by default |
| `vectors.txt` | Standard word2vec text format (portable, human-readable) — fallback |
| `vocab_freq.json` | Word frequency counts (needed for accurate incremental training) |

The server loads `model.bin` if present, otherwise falls back to `vectors.txt`.
