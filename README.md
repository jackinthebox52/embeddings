# Text Embeddings CLI Tool

Fast, multithreaded text embedding generator that follows Unix philosophy. Converts text to vector embeddings with guaranteed input order preservation. Supports multiple embedding methods optimized for different use cases.

## Features

- Multiple specialized embedding methods:
  - TF-IDF: Best for longer texts, documents ([Understanding TF-IDF](https://jonathan.laurent/files/cs/tfidf.pdf))
  - N-gram: Optimized for short texts ([Character n-grams](https://aclanthology.org/P15-1107.pdf))
  - WordPiece: Balanced approach ([Google's WordPiece](https://arxiv.org/pdf/1609.08144.pdf))
  - Soundex: Phonetic similarity, great for names ([Soundex Algorithm](https://en.wikipedia.org/wiki/Soundex))
  - Positional: Preserves word order importance ([Positional Encoding](https://arxiv.org/abs/1706.03762))
  - Keyphrase: Topic and keyword focused ([Keyphrase Extraction](https://aclanthology.org/P10-1065/))
  - Semantic Role: Basic syntax understanding ([Semantic Role Labeling](https://direct.mit.edu/coli/article/45/2/207/93605/A-Survey-on-Deep-Learning-Methods-for-Semantic-Role))
- Preserves input order in output
- Parallel processing with automatic thread count detection
- Normalized output vectors compatible with vector databases

## Method Selection Guide

- **TF-IDF**: Documents, articles, long texts
  - Best for: Document classification, topic modeling
  - Example: Blog posts, academic papers

- **N-gram**: Short texts, character patterns
  - Best for: Product names, short titles
  - Example: SKUs, usernames

- **WordPiece**: Mixed-length texts, subword patterns
  - Best for: General purpose, mixed content
  - Example: Social media posts, product descriptions

- **Soundex**: Phonetic matching
  - Best for: Names, fuzzy matching
  - Example: Customer names, company names
  - Handles: "Smith" ≈ "Smythe" ≈ "Smithe"

- **Positional**: Word order sensitivity
  - Best for: Sentences where order matters
  - Example: Commands, instructions
  - Distinguishes: "dog bites man" vs "man bites dog"

- **Keyphrase**: Topic and keyword focused
  - Best for: Content summarization, topic extraction
  - Example: News articles, documentation
  - Emphasizes: Important phrases, capitalized terms

- **Semantic Role**: Basic syntax understanding
  - Best for: Question answering, information extraction
  - Example: Queries, facts
  - Captures: Who did what to whom, when, where

[Rest of README remains the same...]

## Usage Examples by Task

```bash
# Name matching
echo -e "Smith\nSmythe\nSmithson" | embeddings -method soundex

# Sentence comparison
echo -e "The dog bit the man\nThe man bit the dog" | embeddings -method positional

# Topic extraction
cat article.txt | embeddings -method keyphrase

# Question processing
echo "Who went to Paris last summer?" | embeddings -method semantic
```

[Rest of README remains the same...]
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