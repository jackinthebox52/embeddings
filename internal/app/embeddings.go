package app

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"
)

// Document represents a text and its vector embedding
type Document struct {
	Text     string    `json:"text"`
	Vector   []float64 `json:"vector"`
	Original string    `json:"original"`
	Method   string    `json:"method"`
	id       int       // private field for internal ordering
}

// Run executes the main application logic
func Run() error {
	// Calculate default thread count
	defaultThreads := runtime.NumCPU() / 2
	if defaultThreads < 1 {
		defaultThreads = 1
	}

	config := parseFlags(defaultThreads)

	if *config.Help {
		printHelp()
		return nil
	}

	// Read input documents
	documents, err := readDocuments(config)
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}

	// Process documents
	results, err := processDocuments(documents, config)
	if err != nil {
		return fmt.Errorf("processing documents: %w", err)
	}

	// Output results
	if err := writeResults(results); err != nil {
		return fmt.Errorf("writing results: %w", err)
	}

	return nil
}

func writeResults(results []Document) error {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(results)
}

// processDocuments processes a batch of documents using the specified embedding method
func processDocuments(docs []string, config Config) ([]Document, error) {
	if config.Method == "tfidf" {
		return processTFIDF(docs, config)
	}
	return processParallel(docs, config)
}

func processParallel(docs []string, config Config) ([]Document, error) {
	jobs := make(chan processingJob, len(docs))
	results := make(chan Document, len(docs))
	var wg sync.WaitGroup

	// Start worker pool
	for w := 0; w < *config.Threads; w++ {
		wg.Add(1)
		go worker(jobs, results, &wg, config)
	}

	// Send jobs to workers
	for i, doc := range docs {
		jobs <- processingJob{
			text: doc,
			id:   i,
		}
	}
	close(jobs)

	// Wait for all workers to complete in a separate goroutine
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results in an ordered slice
	processed := make([]Document, len(docs))
	for result := range results {
		processed[result.id] = result
	}

	return processed, nil
}

type processingJob struct {
	text string
	id   int
}

func worker(jobs <-chan processingJob, results chan<- Document, wg *sync.WaitGroup, config Config) {
	defer wg.Done()

	for job := range jobs {
		var vec map[string]float64

		switch config.Method {
		case "ngram":
			vec = nGramEmbedding(job.text, config.NGramSize)
		case "wordpiece":
			vec = wordPieceEmbedding(job.text)
		default:
			vec = make(map[string]float64)
		}

		results <- Document{
			Text:     job.text,
			Vector:   randomProjection(vec, config.VectorSize),
			Original: job.text,
			Method:   config.Method,
			id:       job.id, // internal field, won't appear in JSON
		}
	}
}

func processTFIDF(docs []string, config Config) ([]Document, error) {
	tfidfVecs := computeTFIDFParallel(docs, *config.Threads)
	results := make([]Document, len(docs))

	for i, vec := range tfidfVecs {
		results[i] = Document{
			Text:     docs[i],
			Vector:   randomProjection(vec, config.VectorSize),
			Original: docs[i],
			Method:   "tfidf",
			id:       i,
		}
	}

	return results, nil
}
