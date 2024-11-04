package main

import (
	"log"
	"os"

	"github.com/jackinthebox52/embeddings/internal/app"
)

func main() {
	if err := app.Run(); err != nil {
		log.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}
