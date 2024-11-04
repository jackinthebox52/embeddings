package app

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Character n-gram embedding suitable for short texts
func nGramEmbedding(text string, n int) map[string]float64 {
	vector := make(map[string]float64)
	normalized := strings.ToLower(text)
	runes := []rune(normalized)

	// Add character n-grams
	for i := 0; i <= len(runes)-n; i++ {
		ngram := string(runes[i : i+n])
		vector[ngram]++
	}

	// Add position information for start and end
	startGram := string(runes[:min(n, len(runes))])
	endGram := string(runes[max(0, len(runes)-n):])
	vector["START_"+startGram] += 0.5
	vector["END_"+endGram] += 0.5

	return vector
}

// Word piece tokenization inspired embedding
func wordPieceEmbedding(text string) map[string]float64 {
	vector := make(map[string]float64)
	normalized := strings.ToLower(text)
	words := strings.Fields(normalized)

	for _, word := range words {
		// Add full word
		vector["W_"+word] += 1.0

		// Add subwords
		runes := []rune(word)
		for length := 2; length <= len(runes); length++ {
			for start := 0; start <= len(runes)-length; start++ {
				subword := string(runes[start : start+length])
				vector["SW_"+subword] += 0.5
			}
		}

		// Add character bigrams
		for i := 0; i < len(runes)-1; i++ {
			bigram := string(runes[i : i+2])
			vector["BG_"+bigram] += 0.25
		}
	}

	return vector
}

// Parallel implementation of TF-IDF
func computeTFIDFParallel(docs []string, numWorkers int) []map[string]float64 {
	// Calculate term frequencies and document frequencies in parallel
	type docStats struct {
		wordFreqs map[string]int
		docFreqs  map[string]bool
	}

	var wg sync.WaitGroup
	statsChan := make(chan docStats, len(docs))

	// Process documents in chunks
	docsPerWorker := (len(docs) + numWorkers - 1) / numWorkers

	for i := 0; i < numWorkers; i++ {
		start := i * docsPerWorker
		end := min((i+1)*docsPerWorker, len(docs))

		if start >= len(docs) {
			break
		}

		wg.Add(1)
		go func(docs []string) {
			defer wg.Done()

			for _, doc := range docs {
				stats := docStats{
					wordFreqs: make(map[string]int),
					docFreqs:  make(map[string]bool),
				}

				words := strings.Fields(strings.ToLower(doc))
				for _, word := range words {
					stats.wordFreqs[word]++
					stats.docFreqs[word] = true
				}

				statsChan <- stats
			}
		}(docs[start:end])
	}

	// Wait for all workers and close channel
	go func() {
		wg.Wait()
		close(statsChan)
	}()

	// Collect results
	wordFreqs := make([]map[string]int, len(docs))
	docFreqs := make(map[string]int)

	i := 0
	for stats := range statsChan {
		wordFreqs[i] = stats.wordFreqs
		for word := range stats.docFreqs {
			docFreqs[word]++
		}
		i++
	}

	// Calculate TF-IDF vectors
	tfidfVecs := make([]map[string]float64, len(docs))
	for i := range docs {
		tfidfVecs[i] = make(map[string]float64)
		for word, freq := range wordFreqs[i] {
			tf := float64(freq)
			idf := math.Log(float64(len(docs)) / float64(docFreqs[word]))
			tfidfVecs[i][word] = tf * idf
		}
	}

	return tfidfVecs
}

// Deterministic random projection
func randomProjection(vec map[string]float64, size int) []float64 {
	result := make([]float64, size)

	for word, value := range vec {
		hash := int32(0)
		for _, c := range word {
			hash = hash*31 + int32(c)
		}
		for i := 0; i < size; i++ {
			projValue := float64(hash%(1<<16)) / float64(1<<16)
			result[i] += value * projValue
			hash = hash*1103515245 + 12345
		}
	}

	// Normalize
	magnitude := 0.0
	for _, v := range result {
		magnitude += v * v
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude > 0 {
		for i := range result {
			result[i] /= magnitude
		}
	}
	return result
}

// SoundexEmbedding creates phonetic similarity embeddings
// Good for: Names, fuzzy matching, pronunciation-based search
func soundexEmbedding(text string) map[string]float64 {
	vector := make(map[string]float64)
	words := strings.Fields(strings.ToLower(text))

	for _, word := range words {
		if len(word) == 0 {
			continue
		}

		// Keep first letter
		code := string(word[0])
		prev := '0'

		// Soundex coding
		for _, c := range word[1:] {
			var num byte
			switch c {
			case 'b', 'f', 'p', 'v':
				num = '1'
			case 'c', 'g', 'j', 'k', 'q', 's', 'x', 'z':
				num = '2'
			case 'd', 't':
				num = '3'
			case 'l':
				num = '4'
			case 'm', 'n':
				num = '5'
			case 'r':
				num = '6'
			default:
				num = '0'
			}

			if num != '0' && num != byte(prev) {
				code += string(num)
			}
			prev = rune(num)

			if len(code) >= 4 {
				break
			}
		}

		// Pad with zeros if needed
		for len(code) < 4 {
			code += "0"
		}

		vector["SDX_"+code] += 1.0
	}

	return vector
}

// PositionalEmbedding creates embeddings that preserve word position information
// Good for: Sentence structure, word order significance, syntactic similarity
func positionalEmbedding(text string) map[string]float64 {
	vector := make(map[string]float64)
	words := strings.Fields(strings.ToLower(text))

	for i, word := range words {
		// Absolute position features
		relPos := float64(i) / float64(len(words))
		vector[fmt.Sprintf("W_%s_%.2f", word, relPos)] += 1.0

		// Beginning/end markers with exponential decay
		startWeight := math.Exp(-float64(i))
		endWeight := math.Exp(-float64(len(words) - i - 1))
		vector["START_"+word] += startWeight
		vector["END_"+word] += endWeight

		// Neighboring words (context window)
		for offset := -2; offset <= 2; offset++ {
			if offset == 0 {
				continue
			}
			j := i + offset
			if j >= 0 && j < len(words) {
				vector[fmt.Sprintf("CTX_%d_%s_%s", offset, word, words[j])] += 1.0
			}
		}
	}

	return vector
}

// KeyphraseEmbedding creates embeddings optimized for key phrase extraction
// Good for: Topic detection, keyword extraction, document summarization
func keyphraseEmbedding(text string) map[string]float64 {
	vector := make(map[string]float64)
	words := strings.Fields(strings.ToLower(text))

	// Track phrase frequencies
	phrases := make(map[string]float64)

	// Single words with position and capitalization features
	for i, word := range words {
		// Skip stopwords for phrases
		if isStopword(word) {
			continue
		}

		// Check if word was capitalized in original
		wasCapitalized := false
		if i < len(words) && unicode.IsUpper(rune(words[i][0])) {
			wasCapitalized = true
		}

		// Add single word features
		phrases[word] += 1.0
		if wasCapitalized {
			phrases[word] += 0.5
		}

		// Position bonus for words at start/end
		if i < len(words)/5 { // First 20%
			phrases[word] += 0.3
		}
		if i > len(words)*4/5 { // Last 20%
			phrases[word] += 0.3
		}
	}

	// Extract top phrases by score
	type phraseScore struct {
		phrase string
		score  float64
	}
	var sortedPhrases []phraseScore
	for phrase, score := range phrases {
		sortedPhrases = append(sortedPhrases, phraseScore{phrase, score})
	}
	sort.Slice(sortedPhrases, func(i, j int) bool {
		return sortedPhrases[i].score > sortedPhrases[j].score
	})

	// Add top phrases to vector
	for i, ps := range sortedPhrases {
		if i >= 20 { // Keep top 20 phrases
			break
		}
		vector["KP_"+ps.phrase] = ps.score
	}

	return vector
}

// SemanticRoleEmbedding creates embeddings based on basic semantic role labeling
// Good for: Question answering, information extraction, semantic search
func semanticRoleEmbedding(text string) map[string]float64 {
	vector := make(map[string]float64)
	words := strings.Fields(strings.ToLower(text))

	if len(words) == 0 {
		return vector
	}

	// Simple heuristic-based role labeling
	// Not as sophisticated as ML-based SRL but useful for many cases

	// Detect potential subjects (usually at start)
	for i, word := range words {
		if i > len(words)/3 { // Look in first third
			break
		}
		if !isStopword(word) {
			vector["SUBJ_"+word] += 1.0 - float64(i)/float64(len(words))
		}
	}

	// Detect potential objects (usually after verbs)
	for i := 1; i < len(words); i++ {
		if isVerb(words[i-1]) && !isStopword(words[i]) {
			vector["OBJ_"+words[i]] += 1.0
		}
	}

	// Detect action verbs
	for _, word := range words {
		if isVerb(word) {
			vector["ACTION_"+word] += 1.0
		}
	}

	// Detect time/location indicators
	for i, word := range words {
		if isTimeIndicator(word) {
			vector["TIME_"+word] += 1.0
			// Include next word if not stopword
			if i+1 < len(words) && !isStopword(words[i+1]) {
				vector["TIME_"+word+"_"+words[i+1]] += 0.5
			}
		}
		if isLocationIndicator(word) {
			vector["LOC_"+word] += 1.0
			// Include next word if not stopword
			if i+1 < len(words) && !isStopword(words[i+1]) {
				vector["LOC_"+word+"_"+words[i+1]] += 0.5
			}
		}
	}

	return vector
}

// Helper functions for semantic role labeling
func isStopword(word string) bool {
	stopwords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
	}
	return stopwords[word]
}

func isVerb(word string) bool {
	commonVerbs := map[string]bool{
		"is": true, "are": true, "was": true, "were": true,
		"have": true, "has": true, "had": true,
		"do": true, "does": true, "did": true,
		"go": true, "went": true, "gone": true,
		"make": true, "made": true, "say": true, "said": true,
		"get": true, "got": true, "take": true, "took": true,
	}
	return commonVerbs[word]
}

func isTimeIndicator(word string) bool {
	timeWords := map[string]bool{
		"today": true, "tomorrow": true, "yesterday": true,
		"now": true, "then": true, "when": true,
		"morning": true, "afternoon": true, "evening": true,
		"monday": true, "tuesday": true, "wednesday": true,
		"thursday": true, "friday": true, "saturday": true, "sunday": true,
	}
	return timeWords[word]
}

func isLocationIndicator(word string) bool {
	locationWords := map[string]bool{
		"in": true, "at": true, "on": true, "near": true,
		"under": true, "over": true, "between": true,
		"here": true, "there": true, "where": true,
		"north": true, "south": true, "east": true, "west": true,
	}
	return locationWords[word]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
