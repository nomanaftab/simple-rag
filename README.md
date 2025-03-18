# Multi-Document RAG System with LlamaIndex

This repository contains a flexible and robust Retrieval-Augmented Generation (RAG) system built with LlamaIndex. It supports multiple document types and vector database backends for efficient document indexing and semantic retrieval.

## Features

- **Multi-Document Support**: Extract text from PDFs, Word documents, Excel spreadsheets, PowerPoint presentations, images (via OCR), and plain text files
- **Multiple Vector Database Options**: Use Chroma DB or Qdrant for vector storage
- **Hybrid Search Capability**: Combine vector search with BM25 for better retrieval results
- **Resilient Querying**: Includes fallback mechanisms if primary retrieval methods fail
- **Local LLM Integration**: Uses Ollama for local LLM and embedding generation

## Requirements

- Python 3.9+
- Ollama with `llama3.2` model and `nomic-embed-text` embedding model
- PyTesseract (for OCR capabilities)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llama-index-rag.git
   cd llama-index-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup Ollama

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

## Usage

### 1. Index Documents

Choose one of the indexing scripts based on your preferred vector store:

#### With ChromaDB:
```bash
python index_multidoc_chroma.py
```

#### With Qdrant:
```bash
python index_multidoc_qdrant.py
```

By default, the scripts will:
- Look for documents in a `./data` directory
- Create an index in `./chroma_db` or `./qdrant_db` respectively

### 2. Query the Index

After indexing, you can query your documents using the query scripts:

#### Basic Query with ChromaDB:
```bash
python query_chroma.py
```

#### Advanced Query with ChromaDB:
```bash
python query_adv_chroma.py
```

#### Basic Query with Qdrant:
```bash
python query_qdrant.py
```

#### Advanced Query with Qdrant:
```bash
python query_adv_qdrant.py
```

## Configuring Vector Stores

### ChromaDB Configuration

ChromaDB is used with a persistent client to store vectors on disk. The default configuration in `index_multidoc_chroma.py` is:

```python
chroma_client = chromadb.PersistentClient(path=output_dir)
collection_name = "document_collection"
```

### Qdrant Configuration

Qdrant supports multiple deployment options:

1. **Local Embedding Database**:
   ```python
   client = qdrant_client.QdrantClient(path="./qdrant_db")
   ```

2. **Local Server Mode**:
   ```python
   client = qdrant_client.QdrantClient(url="http://localhost:6333")
   ```

3. **Qdrant Cloud**:
   ```python
   client = qdrant_client.QdrantClient(
       url="https://your-deployment-url.qdrant.io",
       api_key="your-api-key"
   )
   ```

## Customization

### Changing Document Directories

Modify the paths in the main section of the indexing scripts:

```python
# Directory containing documents to index
input_directory = "./your_docs_folder"

# Directory where the index will be stored
output_directory = "./your_db_path"
```

### Adjusting Retrieval Parameters

In the query scripts, you can modify:

- `similarity_top_k`: Number of documents to retrieve
- `vector_store_query_mode`: Set to "mmr" for Maximum Marginal Relevance (more diverse results)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
