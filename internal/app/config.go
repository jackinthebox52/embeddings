package app

import (
	"flag"
	"fmt"
	"os"
)

type Config struct {
	Method     string
	VectorSize int
	NGramSize  int
	Threads    *int
	Help       *bool
	Benchmark  bool
	InputFile  string
}

func parseFlags(defaultThreads int) Config {
	method := flag.String("method", "tfidf", "Embedding method: tfidf, ngram, or wordpiece")
	vectorSize := flag.Int("size", 64, "Size of the output vector")
	ngramSize := flag.Int("ngram-size", 3, "Size of character n-grams (for ngram method)")
	threads := flag.Int("threads", defaultThreads, "Number of worker threads")
	help := flag.Bool("help", false, "Show help message")
	benchmark := flag.Bool("benchmark", false, "Run benchmark and show timing info")
	inputFile := flag.String("input", "", "Input file path (if not specified, reads from stdin)")

	flag.Parse()

	return Config{
		Method:     *method,
		VectorSize: *vectorSize,
		NGramSize:  *ngramSize,
		Threads:    threads,
		Help:       help,
		Benchmark:  *benchmark,
		InputFile:  *inputFile,
	}
}

func printHelp() {
	fmt.Fprintf(os.Stderr, `Text Embedding CLI Tool

Usage: %s [options]

Options:
  -method string
    	Embedding method (default "tfidf")
    	Available methods:
    	- tfidf: Term Frequency-Inverse Document Frequency (best for longer texts)
    	- ngram: Character n-gram based (good for short texts)
    	- wordpiece: Word piece inspired (balanced approach)
  -size int
    	Size of the output vector (default 64)
  -ngram-size int
    	Size of character n-grams for ngram method (default 3)
  -threads int
    	Number of worker threads (default: NumCPU/2)
  -input string
    	Input file path (if not specified, reads from stdin)
  -benchmark
    	Run benchmark and show timing info
  -help
    	Show this help message

Input: Read from -input file if specified, otherwise from stdin.
Output: JSON written to stdout.
`, os.Args[0])
}
