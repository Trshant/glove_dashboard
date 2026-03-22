package query

import (
	"encoding/json"
	"testing"
)

func TestBuildQueryPositiveOnly(t *testing.T) {
	q := BuildQuery([]string{"good", "great"}, nil)

	// Verify structure
	data, err := json.Marshal(q)
	if err != nil {
		t.Fatal(err)
	}

	var result map[string]any
	json.Unmarshal(data, &result)

	if result["from"] != float64(0) {
		t.Errorf("expected from=0, got %v", result["from"])
	}
	if result["size"] != float64(100) {
		t.Errorf("expected size=100, got %v", result["size"])
	}
	if result["track_total_hits"] != true {
		t.Errorf("expected track_total_hits=true, got %v", result["track_total_hits"])
	}

	query := result["query"].(map[string]any)
	boolQ := query["bool"].(map[string]any)
	should := boolQ["should"].([]any)

	if len(should) != 1 {
		t.Fatalf("expected 1 should clause for positive-only, got %d", len(should))
	}

	innerBool := should[0].(map[string]any)["bool"].(map[string]any)
	innerShould := innerBool["should"].([]any)
	if len(innerShould) != 2 {
		t.Errorf("expected 2 positive match phrases, got %d", len(innerShould))
	}
}

func TestBuildQueryWithNegatives(t *testing.T) {
	q := BuildQuery([]string{"good"}, []string{"bad"})

	data, _ := json.Marshal(q)
	var result map[string]any
	json.Unmarshal(data, &result)

	query := result["query"].(map[string]any)
	boolQ := query["bool"].(map[string]any)
	should := boolQ["should"].([]any)

	if len(should) != 2 {
		t.Fatalf("expected 2 should clauses (positive + negative), got %d", len(should))
	}

	// First clause: positive with "should"
	posBool := should[0].(map[string]any)["bool"].(map[string]any)
	if _, ok := posBool["should"]; !ok {
		t.Error("expected 'should' in positive clause")
	}

	// Second clause: negative with "must"
	negBool := should[1].(map[string]any)["bool"].(map[string]any)
	if _, ok := negBool["must"]; !ok {
		t.Error("expected 'must' in negative clause")
	}
}
