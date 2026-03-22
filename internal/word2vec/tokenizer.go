package word2vec

import (
	"regexp"
	"strings"
)

var wordRe = regexp.MustCompile(`\w+`)

// Tokenize extracts lowercase word tokens from text, matching Python's re.findall(r'\w+', text.lower()).
func Tokenize(text string) []string {
	return wordRe.FindAllString(strings.ToLower(text), -1)
}
