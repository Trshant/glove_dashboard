package handlers

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"
	"strings"

	"glove/internal/elastic"
	"glove/internal/query"
	"glove/internal/word2vec"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

// formTermData is the template data for form_term.html.
type formTermData struct {
	Data []wordEntry
}

type wordEntry struct {
	Index int
	Word  string
	Score float64
}

// searchResultsData is the template data for search_results.html.
type searchResultsData struct {
	Query string
	Total int
	Data  []searchHit
}

type searchHit struct {
	ID      string
	Content string
}

// NewRouter creates a chi router with all HTTP handlers.
func NewRouter(model *word2vec.Model, es *elastic.Client, tmpl *template.Template) chi.Router {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	h := &handler{
		model: model,
		es:    es,
		tmpl:  tmpl,
	}

	r.Get("/", h.index)
	r.Post("/submit-form", h.submitForm)
	r.Post("/submit-search", h.submitSearch)

	return r
}

type handler struct {
	model *word2vec.Model
	es    *elastic.Client
	tmpl  *template.Template
}

// GET / — render index.html
func (h *handler) index(w http.ResponseWriter, r *http.Request) {
	if err := h.tmpl.ExecuteTemplate(w, "index.html", nil); err != nil {
		log.Printf("rendering index: %v", err)
		http.Error(w, "Internal Server Error", 500)
	}
}

// POST /submit-form — get similar words for the submitted term
func (h *handler) submitForm(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad Request", 400)
		return
	}

	name := strings.TrimSpace(r.FormValue("name"))
	if name == "" {
		http.Error(w, "name is required", 400)
		return
	}

	words := word2vec.Tokenize(name)
	similar := h.model.GetSimilarWords(words, nil, 15, true)

	data := formTermData{}
	for i, ws := range similar {
		data.Data = append(data.Data, wordEntry{
			Index: i,
			Word:  ws.Word,
			Score: ws.Score,
		})
	}

	if err := h.tmpl.ExecuteTemplate(w, "form_term.html", data); err != nil {
		log.Printf("rendering form_term: %v", err)
		http.Error(w, "Internal Server Error", 500)
	}
}

// POST /submit-search — search ES with positive/negative terms
func (h *handler) submitSearch(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "Bad Request", 400)
		return
	}

	positive := r.Form["pro[]"]
	negative := r.Form["con[]"]

	if len(positive) == 0 && len(negative) == 0 {
		// Nothing selected, return empty
		w.Write([]byte(""))
		return
	}

	esQuery := query.BuildQuery(positive, negative)

	// Pretty-print query for display
	queryJSON, _ := json.MarshalIndent(esQuery, "", "  ")

	result, err := h.es.Search(esQuery)
	if err != nil {
		log.Printf("search error: %v", err)
		http.Error(w, "Search failed", 500)
		return
	}

	data := searchResultsData{
		Query: string(queryJSON),
		Total: result.Total,
	}
	for _, hit := range result.Hits {
		data.Data = append(data.Data, searchHit{
			ID:      hit.ID,
			Content: hit.Content,
		})
	}

	if err := h.tmpl.ExecuteTemplate(w, "search_results.html", data); err != nil {
		log.Printf("rendering search_results: %v", err)
		http.Error(w, "Internal Server Error", 500)
	}
}
