package app

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Document represents a text and its vector embedding
type InputStats struct {
	DocumentCount  int           `json:"document_count"`
	TotalBytes     int           `json:"total_bytes"`
	AvgDocLength   float64       `json:"avg_doc_length"`
	MinDocLength   int           `json:"min_doc_length"`
	MaxDocLength   int           `json:"max_doc_length"`
	ProcessingTime time.Duration `json:"processing_time"`
}

type EmbeddingStats struct {
	Method          string        `json:"method"`
	VectorSize      int           `json:"vector_size"`
	TotalTime       time.Duration `json:"total_time"`
	TimePerDocument time.Duration `json:"time_per_document"`
	DocsPerSecond   float64       `json:"docs_per_second"`
	ThreadsUsed     int           `json:"threads_used"`
}

type MemoryStats struct {
	PeakMemory   uint64        `json:"peak_memory_bytes"`
	AllocatedMem uint64        `json:"allocated_memory_bytes"`
	NumGC        uint32        `json:"num_gc_cycles"`
	GCPauseTotal time.Duration `json:"gc_pause_total"`
}

type VectorStats struct {
	AvgMagnitude   float64 `json:"avg_magnitude"`
	AvgSparsity    float64 `json:"avg_sparsity"`
	UniqueFeatures int     `json:"unique_features"`
	TotalFeatures  int     `json:"total_features"`
}

type BenchmarkStats struct {
	InputStats     InputStats     `json:"input_stats"`
	EmbeddingStats EmbeddingStats `json:"embedding_stats"`
	MemoryStats    MemoryStats    `json:"memory_stats"`
	VectorStats    VectorStats    `json:"vector_stats"`
}

type Document struct {
	Text     string    `json:"text"`
	Vector   []float64 `json:"vector"`
	Original string    `json:"original"`
	Method   string    `json:"method"`
	id       int       // private field for internal ordering (not exported to JSON)
}

// Run executes the main application logic
func Run() error {
	config := parseFlags(runtime.NumCPU() / 2)

	if *config.Help {
		printHelp()
		return nil
	}

	var stats BenchmarkStats
	var err error

	// Capture start memory stats
	var memStatsBefore runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	// Time entire process
	processStart := time.Now()

	// Read and analyze input
	documents, err := readDocuments(config)
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}

	inputStats := analyzeInput(documents)
	stats.InputStats = inputStats
	stats.InputStats.ProcessingTime = time.Since(processStart)

	// Process documents with timing
	embedStart := time.Now()
	results, err := processDocuments(documents, config)
	if err != nil {
		return fmt.Errorf("processing documents: %w", err)
	}
	embedTime := time.Since(embedStart)

	// Capture end memory stats
	var memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsAfter)

	// Collect embedding stats
	stats.EmbeddingStats = collectEmbeddingStats(config, documents, embedTime)
	stats.MemoryStats = collectMemoryStats(memStatsBefore, memStatsAfter)
	stats.VectorStats = analyzeVectors(results)

	// Write results
	if err := writeResults(results); err != nil {
		return fmt.Errorf("writing results: %w", err)
	}

	// Output benchmark results if requested
	if config.Benchmark {
		return outputBenchmark(stats)
	}

	return nil
}

func outputBenchmark(stats BenchmarkStats) error {
	// Create formatted benchmark report
	var report strings.Builder

	report.WriteString("\n=== Embedding Benchmark Report ===\n\n")

	// Input Statistics
	report.WriteString("Input Statistics:\n")
	report.WriteString(fmt.Sprintf("  Documents processed: %d\n", stats.InputStats.DocumentCount))
	report.WriteString(fmt.Sprintf("  Total input size: %.2f MB\n", float64(stats.InputStats.TotalBytes)/1024/1024))
	report.WriteString(fmt.Sprintf("  Average document length: %.1f bytes\n", stats.InputStats.AvgDocLength))
	report.WriteString(fmt.Sprintf("  Document length range: %d - %d bytes\n", stats.InputStats.MinDocLength, stats.InputStats.MaxDocLength))
	report.WriteString(fmt.Sprintf("  Input processing time: %v\n\n", stats.InputStats.ProcessingTime))

	// Embedding Performance
	report.WriteString("Embedding Performance:\n")
	report.WriteString(fmt.Sprintf("  Method: %s\n", stats.EmbeddingStats.Method))
	report.WriteString(fmt.Sprintf("  Vector size: %d\n", stats.EmbeddingStats.VectorSize))
	report.WriteString(fmt.Sprintf("  Total processing time: %v\n", stats.EmbeddingStats.TotalTime))
	report.WriteString(fmt.Sprintf("  Average time per document: %v\n", stats.EmbeddingStats.TimePerDocument))
	report.WriteString(fmt.Sprintf("  Documents per second: %.1f\n", stats.EmbeddingStats.DocsPerSecond))
	report.WriteString(fmt.Sprintf("  Threads used: %d\n\n", stats.EmbeddingStats.ThreadsUsed))

	// Memory Usage
	report.WriteString("Memory Statistics:\n")
	report.WriteString(fmt.Sprintf("  Peak memory usage: %.2f MB\n", float64(stats.MemoryStats.PeakMemory)/1024/1024))
	report.WriteString(fmt.Sprintf("  Final allocated memory: %.2f MB\n", float64(stats.MemoryStats.AllocatedMem)/1024/1024))
	report.WriteString(fmt.Sprintf("  GC cycles: %d\n", stats.MemoryStats.NumGC))
	report.WriteString(fmt.Sprintf("  Total GC pause time: %v\n\n", stats.MemoryStats.GCPauseTotal))

	// Vector Statistics
	report.WriteString("Vector Statistics:\n")
	report.WriteString(fmt.Sprintf("  Average vector magnitude: %.4f\n", stats.VectorStats.AvgMagnitude))
	report.WriteString(fmt.Sprintf("  Average vector sparsity: %.2f%%\n", stats.VectorStats.AvgSparsity*100))
	report.WriteString(fmt.Sprintf("  Unique features: %d\n", stats.VectorStats.UniqueFeatures))
	report.WriteString(fmt.Sprintf("  Total features: %d\n\n", stats.VectorStats.TotalFeatures))

	// Print human-readable report to stderr
	fmt.Fprintf(os.Stderr, "%s", report.String())

	// Write detailed JSON stats to separate file if requested
	if jsonStats, err := json.MarshalIndent(stats, "", "  "); err == nil {
		statsFile := fmt.Sprintf("embedding_benchmark_%s_%d.json",
			stats.EmbeddingStats.Method,
			time.Now().Unix())
		if err := os.WriteFile(statsFile, jsonStats, 0644); err != nil {
			return fmt.Errorf("writing benchmark stats: %w", err)
		}
		fmt.Fprintf(os.Stderr, "Detailed benchmark stats written to: %s\n", statsFile)
	}

	return nil
}

func analyzeInput(documents []string) InputStats {
	stats := InputStats{}

	stats.DocumentCount = len(documents)
	minLen := -1
	maxLen := 0
	totalLen := 0

	for _, doc := range documents {
		length := len(doc)
		totalLen += length
		stats.TotalBytes += length

		if minLen == -1 || length < minLen {
			minLen = length
		}
		if length > maxLen {
			maxLen = length
		}
	}

	stats.MinDocLength = minLen
	stats.MaxDocLength = maxLen
	if stats.DocumentCount > 0 {
		stats.AvgDocLength = float64(totalLen) / float64(stats.DocumentCount)
	}

	return stats
}

func collectEmbeddingStats(config Config, documents []string, duration time.Duration) EmbeddingStats {
	stats := EmbeddingStats{
		Method:      config.Method,
		VectorSize:  config.VectorSize,
		TotalTime:   duration,
		ThreadsUsed: *config.Threads,
	}

	docCount := float64(len(documents))
	if docCount > 0 {
		stats.TimePerDocument = time.Duration(int64(duration) / int64(docCount))
		stats.DocsPerSecond = docCount / duration.Seconds()
	}

	return stats
}

func collectMemoryStats(before, after runtime.MemStats) MemoryStats {
	return MemoryStats{
		PeakMemory:   after.TotalAlloc,
		AllocatedMem: after.Alloc,
		NumGC:        after.NumGC - before.NumGC,
		GCPauseTotal: time.Duration(after.PauseTotalNs - before.PauseTotalNs),
	}
}

func analyzeVectors(results []Document) VectorStats {
	stats := VectorStats{}

	uniqueFeatures := make(map[string]struct{})
	totalFeatures := 0

	for _, doc := range results {
		magnitude := 0.0
		nonZeroCount := 0
		for _, val := range doc.Vector {
			magnitude += val * val
			if val != 0 {
				nonZeroCount++
			}
		}
		stats.AvgMagnitude += magnitude

		sparsity := 1.0 - float64(nonZeroCount)/float64(len(doc.Vector))
		stats.AvgSparsity += sparsity

		totalFeatures += nonZeroCount
	}

	docCount := float64(len(results))
	if docCount > 0 {
		stats.AvgMagnitude = stats.AvgMagnitude / docCount
		stats.AvgSparsity = stats.AvgSparsity / docCount
	}
	stats.UniqueFeatures = len(uniqueFeatures)
	stats.TotalFeatures = totalFeatures

	return stats
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
