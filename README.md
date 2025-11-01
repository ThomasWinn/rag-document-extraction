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

### 1️⃣ Put the Entire RFP in Context (No RAG)
You literally feed the whole document (e.g., 30-page PDF) into the model’s context window.

#### Pros
- ✅ No retrieval errors — the model sees everything at once.
- ✅ Works fine for short documents (a few pages).
- ✅ Easier to set up — just one prompt.

#### Cons
- ⚠️ Context-length limits: most models can’t handle 100k+ tokens efficiently.
- ⚠️ Higher latency and cost (every token counts).
- ⚠️ Model might get “distracted” — it sees too much and misses key details.
- ⚠️ Answers degrade for long documents — LLMs don’t perfectly remember the start of a huge context.

👉 **Best for:** Quick tests, small PDFs (under 10 pages), or tasks where accuracy isn’t mission-critical.

### 2️⃣ Use RAG (Retrieve → Augment → Generate)
You split the RFP into chunks (paragraphs, sections, etc.), store them in a vector database, and retrieve only the most relevant parts at query time.

#### Pros
- ✅ Scales to any document size — even hundreds of pages.
- ✅ Faster, cheaper, more memory-efficient.
- ✅ Keeps responses focused — model only sees what matters.
- ✅ Easier to trace (“Here’s the source paragraph that answered your question”).

#### Cons
- ⚠️ If your chunking or embeddings aren’t tuned, retrieval might miss the relevant section.
- ⚠️ Context is limited to what’s retrieved — if the right text isn’t pulled, the model can’t know it.

👉 **Best for:** Large, structured RFPs or collections of proposals — especially if you’ll be asking many different questions (pricing, benefits, eligibility, etc.) across many docs.
