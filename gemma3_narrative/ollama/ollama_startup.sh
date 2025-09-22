#!/bin/bash
# Script to ensure models are available before starting the server

echo "Starting Ollama model pull and serve sequence..."

# Pull Gemma 3B
ollama pull gemma:3b

# Pull Deepseek 7B
ollama pull deepseek-llm:7b

echo "Model downloads complete. Starting Ollama server..."

# Start the Ollama server and keep it running
OLLAMA_HOST=0.0.0.0 ollama serve
