package app

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

// readDocuments reads input from either a file or stdin based on config
func readDocuments(config Config) ([]string, error) {
	var reader io.Reader

	if config.InputFile != "" {
		file, err := os.Open(config.InputFile)
		if err != nil {
			return nil, fmt.Errorf("opening input file: %w", err)
		}
		defer file.Close()
		reader = file
	} else {
		// Check if stdin is empty
		stat, err := os.Stdin.Stat()
		if err != nil {
			return nil, fmt.Errorf("checking stdin: %w", err)
		}

		if (stat.Mode() & os.ModeCharDevice) != 0 {
			return nil, fmt.Errorf("no input provided: pipe input or use -input flag")
		}
		reader = os.Stdin
	}

	return readFromReader(reader)
}

func readFromReader(reader io.Reader) ([]string, error) {
	var documents []string
	scanner := bufio.NewScanner(reader)

	// Increase scanner buffer size for large lines
	const maxCapacity = 512 * 1024 // 512KB
	buf := make([]byte, maxCapacity)
	scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		text := scanner.Text()
		if text != "" {
			documents = append(documents, text)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading input: %w", err)
	}

	if len(documents) == 0 {
		return nil, fmt.Errorf("no documents found in input")
	}

	return documents, nil
}
