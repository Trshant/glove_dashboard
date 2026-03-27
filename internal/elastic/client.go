package elastic

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/esapi"
)

// Client wraps the Elasticsearch client for index management and search.
type Client struct {
	es        *elasticsearch.Client
	indexName string
}

// SearchHit represents a single search result.
type SearchHit struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

// SearchResult holds search results and metadata.
type SearchResult struct {
	Hits  []SearchHit `json:"hits"`
	Total int         `json:"total"`
}

// NewClient creates an Elasticsearch client with basic auth.
func NewClient(addr, user, pass, indexName string) (*Client, error) {
	cfg := elasticsearch.Config{
		Addresses: []string{addr},
		Username:  user,
		Password:  pass,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}

	es, err := elasticsearch.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating ES client: %w", err)
	}

	c := &Client{es: es, indexName: indexName}
	if err := c.EnsureIndexExists(); err != nil {
		return nil, err
	}
	return c, nil
}

// EnsureIndexExists creates the index if it doesn't already exist.
func (c *Client) EnsureIndexExists() error {
	res, err := c.es.Indices.Exists([]string{c.indexName})
	if err != nil {
		return fmt.Errorf("checking index existence: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode == 200 {
		return nil // Index already exists
	}

	return c.createIndex()
}

func (c *Client) createIndex() error {
	settings := map[string]any{
		"settings": map[string]any{
			"index": map[string]any{
				"max_result_window": 50000,
				"similarity": map[string]any{
					"content_similarity": map[string]any{
						"type": "BM25",
					},
				},
			},
		},
		"mappings": map[string]any{
			"properties": map[string]any{
				"content": map[string]any{
					"type":       "text",
					"similarity": "content_similarity",
				},
			},
		},
	}

	body, err := json.Marshal(settings)
	if err != nil {
		return err
	}

	res, err := c.es.Indices.Create(c.indexName, c.es.Indices.Create.WithBody(bytes.NewReader(body)))
	if err != nil {
		return fmt.Errorf("creating index: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		b, _ := io.ReadAll(res.Body)
		return fmt.Errorf("creating index: %s", b)
	}

	return nil
}

// InsertDocument indexes a single document with the given content.
func (c *Client) InsertDocument(content string) error {
	doc := map[string]string{"content": content}
	body, err := json.Marshal(doc)
	if err != nil {
		return err
	}

	res, err := c.es.Index(c.indexName, bytes.NewReader(body),
		c.es.Index.WithRefresh("true"))
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.IsError() {
		b, _ := io.ReadAll(res.Body)
		return fmt.Errorf("indexing document: %s", b)
	}
	return nil
}

// BulkInsert indexes documents in batches for performance.
// onProgress is called after each batch with the total documents indexed so far.
func (c *Client) BulkInsert(docs []string, batchSize int, onProgress func(int)) error {
	if batchSize <= 0 {
		batchSize = 500
	}

	for i := 0; i < len(docs); i += batchSize {
		end := i + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[i:end]

		var buf bytes.Buffer
		for _, content := range batch {
			// Action line
			buf.WriteString(`{"index":{}}`)
			buf.WriteByte('\n')
			// Document line
			doc, _ := json.Marshal(map[string]string{"content": content})
			buf.Write(doc)
			buf.WriteByte('\n')
		}

		res, err := c.es.Bulk(bytes.NewReader(buf.Bytes()),
			c.es.Bulk.WithIndex(c.indexName))
		if err != nil {
			return fmt.Errorf("bulk insert: %w", err)
		}
		res.Body.Close()

		if res.IsError() {
			return fmt.Errorf("bulk insert error: %s", res.Status())
		}

		if onProgress != nil {
			onProgress(end)
		}
	}

	// Final refresh
	res, err := c.es.Indices.Refresh(c.es.Indices.Refresh.WithIndex(c.indexName))
	if err != nil {
		return err
	}
	defer res.Body.Close()
	return nil
}

// DocCount returns the number of documents in the index.
func (c *Client) DocCount() (int, error) {
	res, err := c.es.Count(c.es.Count.WithIndex(c.indexName))
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()

	var result struct {
		Count int `json:"count"`
	}
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		return 0, err
	}
	return result.Count, nil
}

// ScrollAll retrieves all documents from the index using the scroll API.
// onBatch is called with each batch of content strings.
func (c *Client) ScrollAll(batchSize int, onBatch func(docs []string)) error {
	if batchSize <= 0 {
		batchSize = 1000
	}

	query := map[string]any{
		"query": map[string]any{"match_all": map[string]any{}},
		"size":  batchSize,
		"_source": []string{"content"},
	}
	body, err := json.Marshal(query)
	if err != nil {
		return err
	}

	// Initial search with scroll
	res, err := c.es.Search(
		c.es.Search.WithIndex(c.indexName),
		c.es.Search.WithBody(bytes.NewReader(body)),
		c.es.Search.WithScroll(scrollTimeout),
	)
	if err != nil {
		return fmt.Errorf("scroll init: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		b, _ := io.ReadAll(res.Body)
		return fmt.Errorf("scroll init error: %s", b)
	}

	var scrollResult scrollResponse
	if err := json.NewDecoder(res.Body).Decode(&scrollResult); err != nil {
		return fmt.Errorf("decoding scroll response: %w", err)
	}

	for {
		docs := extractContent(scrollResult.Hits.Hits)
		if len(docs) == 0 {
			break
		}
		onBatch(docs)

		// Fetch next page
		scrollBody, _ := json.Marshal(map[string]string{
			"scroll":    "2m",
			"scroll_id": scrollResult.ScrollID,
		})
		res, err := c.es.Scroll(
			c.es.Scroll.WithBody(bytes.NewReader(scrollBody)),
		)
		if err != nil {
			return fmt.Errorf("scroll: %w", err)
		}

		scrollResult = scrollResponse{}
		if err := json.NewDecoder(res.Body).Decode(&scrollResult); err != nil {
			res.Body.Close()
			return fmt.Errorf("decoding scroll page: %w", err)
		}
		res.Body.Close()
	}

	// Clear scroll
	if scrollResult.ScrollID != "" {
		clearBody, _ := json.Marshal(map[string][]string{
			"scroll_id": {scrollResult.ScrollID},
		})
		res, err := c.es.ClearScroll(
			c.es.ClearScroll.WithBody(bytes.NewReader(clearBody)),
		)
		if err == nil {
			res.Body.Close()
		}
	}

	return nil
}

const scrollTimeout = 2 * time.Minute

type scrollResponse struct {
	ScrollID string `json:"_scroll_id"`
	Hits     struct {
		Hits []struct {
			Source map[string]any `json:"_source"`
		} `json:"hits"`
	} `json:"hits"`
}

func extractContent(hits []struct {
	Source map[string]any `json:"_source"`
}) []string {
	var docs []string
	for _, h := range hits {
		if c, ok := h.Source["content"].(string); ok && c != "" {
			docs = append(docs, c)
		}
	}
	return docs
}

// Search executes a search query and returns parsed results.
func (c *Client) Search(query map[string]any) (*SearchResult, error) {
	body, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	req := esapi.SearchRequest{
		Index: []string{c.indexName},
		Body:  bytes.NewReader(body),
	}

	res, err := req.Do(context.Background(), c.es)
	if err != nil {
		return nil, fmt.Errorf("search request: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		b, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("search error: %s", b)
	}

	var result struct {
		Hits struct {
			Total struct {
				Value int `json:"value"`
			} `json:"total"`
			Hits []struct {
				ID     string         `json:"_id"`
				Source map[string]any `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding search response: %w", err)
	}

	sr := &SearchResult{
		Total: result.Hits.Total.Value,
	}
	for _, hit := range result.Hits.Hits {
		content, _ := hit.Source["content"].(string)
		sr.Hits = append(sr.Hits, SearchHit{
			ID:      hit.ID,
			Content: content,
		})
	}

	return sr, nil
}
