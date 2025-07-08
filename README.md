# ollamaagent

## Overview

This repository demonstrates a minimal retrieval-augmented generation (RAG) workflow
using [Ollama](https://github.com/ollama/ollama) with LangChain. The `main.py`
script loads text documents, creates embeddings with Ollama, indexes them in a
local Chroma database, and answers questions using a chat model.

## Installation

1. Clone this repository and change into the project directory.
2. Start the Ollama Docker container so the API is available at
   `http://localhost:11434`:

   ```bash
   docker run -d -p 11434:11434 ollama/ollama
   ```

3. Install the required Python packages (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your documentation in a file named `documentacao.txt` in the project
   root. The file will be loaded by `main.py` to generate embeddings.
2. Run the script:

   ```bash
   python main.py
   ```

   The example question in the script is `"Como cadastrar um cliente na plataforma?"`.
   You can modify the question or extend the logic as needed.

## Example Commands

```bash
# start Ollama in Docker
docker run -d -p 11434:11434 ollama/ollama

# install dependencies
pip install -r requirements.txt

# supply your document and run
echo "Meu texto" > documentacao.txt
python main.py
```
