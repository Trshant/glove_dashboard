package word2vec

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello World", []string{"hello", "world"}},
		{"it's a test!", []string{"it", "s", "a", "test"}},
		{"  spaces  ", []string{"spaces"}},
		{"UPPER lower MiXeD", []string{"upper", "lower", "mixed"}},
		{"word123 456", []string{"word123", "456"}},
		{"", nil},
	}

	for _, tt := range tests {
		got := Tokenize(tt.input)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Tokenize(%q) = %v, want %v", tt.input, got, tt.want)
		}
	}
}
