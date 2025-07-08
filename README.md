# ollamaagent

## Overview

This repository demonstrates a minimal retrieval-augmented generation (RAG) workflow
using [Ollama](https://github.com/ollama/ollama) with LangChain. The `main.py`
script loads text documents, creates embeddings with Ollama, indexes them in a
local Chroma database, and answers questions using a chat model. Both the
embedding generation and chat use the `deepseek-r1:8b` model served by Ollama.
Ollama-specific integrations are provided by the `langchain-ollama` package.
The retrieval chain in `main.py` is queried using the `invoke()` method.

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

   The requirements include `langchain-ollama` which provides the
   `OllamaEmbeddings` and `ChatOllama` classes used in `main.py`.

## Usage

1. Place your documentation in a file named `documentacao.txt` in the project
   root (or point the script at another file with `--document`). The file can be
   plain text or a PDF. A small sample is provided in this repository to
   demonstrate how the RAG pipeline works:

   ```
   Como cadastrar um cliente na plataforma:
   1. Acesse o menu "Clientes".
   2. Clique em "Novo Cliente".
   3. Preencha os dados solicitados.
   4. Salve para concluir o cadastro.
   ```

   Replace the example text with your own documentation if desired. The sample
   is intentionally brief so it can be indexed quickly during testing.
2. Run the script. You can optionally pass the path to the documentation file
   and the question to ask:

   ```bash
   python main.py --document documentacao.txt --question "Como cadastrar um cliente na plataforma?"
   ```

   If omitted, both arguments default to the above values.

## Example Commands

```bash
# start Ollama in Docker
docker run -d -p 11434:11434 ollama/ollama

# install dependencies
pip install -r requirements.txt

# supply your document and run with defaults
echo "Meu texto" > documentacao.txt
python main.py

# custom document and question (PDF or text)
python main.py --document docs.pdf --question "Qual \u00e9 o hor\u00e1rio de suporte?"
```

## Web Server

A small Flask application is included to query the RAG pipeline through a web
interface. After installing the requirements and starting Ollama, run:

```bash
python server.py
```

The server will start on [http://localhost:8000](http://localhost:8000). Open
this URL in a browser and submit questions using the page. Answers returned by
the retrieval chain will be displayed below the form.
