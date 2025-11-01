# rag-document-extraction
Ingest in a pdf insurance document and Q&amp;A on questions regarding what's in the doc with citations.

### Tech Stack:
- ChromaDB
- Docling
- DuckDB

### Models
- Embeddings: bge-large-en-v1.5 (1024-d)
- ReRanker: bge-reranker-base
- Inference: Qwen2.5-32B-Instruct-4bit

### Questions:
- What embedding model to use?
- What chunking strategy?
- How do I pick what LLM to do inference with?