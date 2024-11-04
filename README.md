# Text Embeddings CLI Tool

Fast, multithreaded text embedding generator that follows Unix philosophy. Converts text to vector embeddings with guaranteed input order preservation. Supports multiple embedding methods optimized for different text lengths.

## Features

- Multiple embedding methods:
  - TF-IDF: Best for longer texts, documents ([Understanding TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Definition))
  - N-gram: Optimized for short texts ([Character n-grams](https://aclanthology.org/P15-1107.pdf))
  - WordPiece-inspired: Balanced approach ([Google's WordPiece paper](https://arxiv.org/pdf/1609.08144.pdf))
- Preserves input order in output
- Parallel processing
- Normalized output vectors compatible with vector databases

## Installation

```bash
go install github.com/jackinthebox52/embeddings/cmd/embeddings@latest
```

Or build from source:
```bash
git clone github.com/jackinthebox52/embeddings
cd embeddings
go build ./cmd/embeddings
```

## Usage

```bash
# Basic usage (read from stdin)
echo "Hello world\nThis is a test\nAnd this is a third line" >> texts.txt

cat texts.txt | embeddings -method tfidf

# Read from file
embeddings -input texts.txt

# Choose embedding method
embeddings -method ngram -input texts.txt

# Customize processing
embeddings -method wordpiece -threads 8 -size 128 -input texts.txt
```

### Options

```
-method string
      Embedding method: tfidf, ngram, or wordpiece (default "tfidf")
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
```

### Method Selection Guide

- **TF-IDF**: Best for documents, articles, long texts. Most computationally intensive.
- **N-gram**: Ideal for short texts, product names, titles. Fastest method.
- **WordPiece**: Good all-rounder. Works well with mixed-length texts.

## Output Format

Output is a JSON array with elements in the same order as input lines:

```json
[
  {
    "text": "First line",
    "vector": [0.1, -0.2, 0.3, ...],
    "original": "First line",
    "method": "tfidf"
  },
  {
    "text": "Second line",
    "vector": [-0.2, 0.1, -0.3, ...],
    "original": "Second line",
    "method": "tfidf"
  }
]
```

The tool guarantees that:
1. Output array order matches input line order exactly
2. Each vector is normalized to unit length
3. Vector dimensions are consistent within output

## Performance

- Automatic thread count detection for optimal performance
- Memory-efficient streaming input processing
- Order preservation has no performance impact
- Benchmark mode available (`-benchmark` flag)

## Example Pipelines

```bash
# Generate embeddings for a word list
cat words.txt | embeddings -method ngram > vectors.json

# Process multiple files while maintaining order
cat file1.txt file2.txt | embeddings -method wordpiece > combined_vectors.json

# Benchmark different methods
cat corpus.txt | embeddings -method tfidf -benchmark
cat corpus.txt | embeddings -method ngram -benchmark
```

## Contributing

Pull requests welcome! Please ensure you:
1. Add tests for new features
2. Run `go fmt` and `go vet`
3. Verify tests pass: `go test ./...`
4. Maintain input order guarantees

## License

MIT