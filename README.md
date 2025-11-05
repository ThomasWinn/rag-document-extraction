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

## Latest Run Summary
- Targeted 12 in-force attributes using the Docling-driven hierarchical chunking pipeline plus Qwen2.5-32B-Instruct for generation.
- Only 2 of 12 attributes were populated correctly; 10 returned empty or unusable results, indicating the current retrieval windows miss key evidence.
- Chunking approach today: Docling preserves layout → sections aggregated hierarchically → RecursiveCharacter splitter (512/40) → Chroma + bge-large embeddings → reranked with bge-reranker.
- The mix of small, metadata-rich chunks appears to fragment key benefit tables; evidence often lands adjacent to questions, so retrieval fails even with reranking.

## Next Steps on Chunking
- Revisit chunk construction: experiment with product-level slabs (e.g., group all LTD content into a single chunk) so RAG has broader context per attribute query.
- Compare hierarchical spans vs product-level chunks for precision/recall on attribute extraction before adding more heuristics.
- Investigate hybrid retrieval (per-product chunk + smaller supporting snippets) instead of relying solely on fine-grained hierarchical chunks.
- Hold off on “full document in prompt” strategy until we benchmark a refined chunking approach; current experience reinforces that better chunking beats 30-page prompts for attribute extraction.
