"""
Docling-powered ingestion pipeline for RFP / proposal documents.

The goal is to ingest documents exclusively through Docling so that the original
layout (especially tables) is preserved before downstream RAG chunking.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from chromadb import PersistentClient

from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionStatus
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class DoclingIngestionPipeline:
    """
    Ingestion pipeline that parses documents with Docling and emits clean text
    and table chunks ready for retrieval indexing.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        data_path: Path,
        embedding_model_name: str,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        embedding_device: Optional[str] = "mps",
        reranker_model_name: Optional[str] = "BAAI/bge-reranker-base",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_path = Path(data_path)
        self.use_cache = use_cache
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.reranker_model_name = reranker_model_name
        self.embedder: SentenceTransformer = SentenceTransformer(
            embedding_model_name,
            device=self.embedding_device,
            trust_remote_code=True,
        )
        self._reranker: Optional[CrossEncoder] = None # (query , chunk) -> score

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}"
            )
        if not self.data_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory for data path, got: {self.data_path}"
            )

        if self.use_cache:
            self.cache_dir = (
                Path(cache_dir)
                if cache_dir is not None
                else self.data_path / ".docling_cache"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Docling converter configured with defaults (tables preserved by default).
        self.converter = DocumentConverter()
        self.document_class = LCDocument

        # Text splitter for chunking documents. Break up by paragraphs, then lines
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def convert_pdf_documents(
        self, refresh_cache: bool = False
    ) -> Dict[str, DoclingDocument]:
        """
        Convert every PDF in the configured data directory.

        Returns a mapping of filename -> DoclingDocument so downstream callers
        can inspect the structured representation (including tables) without
        re-running OCR.
        """

        documents: Dict[str, DoclingDocument] = {}
        for pdf_path in sorted(self.data_path.rglob("*.pdf")):
            print(f"Processing: {pdf_path}")
            if self.use_cache and not refresh_cache:
                cached_doc = self._load_cached_document(pdf_path)
                if cached_doc is not None:
                    documents[pdf_path.name] = cached_doc
                    print("  Loaded from cache.")
                    continue

            try:
                conversion = self.converter.convert(pdf_path)
            except Exception as exc:  # pragma: no cover - converter failure
                logger.warning(
                    "Docling conversion failed for %s: %s", pdf_path, exc
                )
                continue
            if conversion.status not in {
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            }:
                logger.warning(
                    "Skipping %s due to conversion status: %s",
                    pdf_path,
                    conversion.status,
                )
                continue
            doc = conversion.document
            documents[pdf_path.name] = doc
            if self.use_cache:
                self._store_cached_document(pdf_path, doc)
        return documents

    def _cache_path(self, pdf_path: Path) -> Path:
        if not self.cache_dir:
            raise ValueError("Cache directory not configured.")
        return self.cache_dir / f"{pdf_path.stem}.docling.json"

    def _load_cached_document(self, pdf_path: Path) -> Optional[DoclingDocument]:
        if not self.cache_dir:
            return None

        cache_path = self._cache_path(pdf_path)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_mtime = payload.get("source_mtime")
            if cached_mtime is not None:
                current_mtime = pdf_path.stat().st_mtime
                if current_mtime > float(cached_mtime):
                    return None

            document_payload = payload.get("document")
            if not document_payload:
                return None

            return DoclingDocument.model_validate(document_payload)
        except Exception as exc:  # pragma: no cover - cache decode failure
            logger.warning("Failed to load cache for %s: %s", pdf_path, exc)
            return None

    def _store_cached_document(self, pdf_path: Path, document: DoclingDocument) -> None:
        if not self.cache_dir:
            return

        cache_path = self._cache_path(pdf_path)
        payload = {
            "source_path": str(pdf_path.resolve()),
            "source_mtime": pdf_path.stat().st_mtime,
            "document": document.export_to_dict(),
        }
        try:
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - cache write failure
            logger.warning("Failed to write cache for %s: %s", pdf_path, exc)

    def build_hierarchical_chunks(
        self,
        doc_map: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Convert PDF documents (if not provided) and generate LangChain-ready chunks.

        If `doc_map` is supplied, it should map filenames to `DoclingDocument`
        instances (or full `ConversionResult` objects). The chunks retain section
        hierarchy metadata such as section title, path, and page numbers to support
        structured retrieval workflows.
        """

        if doc_map is None:
            doc_map = self.convert_pdf_documents()

        chunk_map: Dict[str, List[DocumentChunk]] = {}

        # pdf path : DoclingDocument
        for filename, doc in doc_map.items():
            doc_id = Path(filename).stem
            sections = self._build_sections(doc, doc_id)

            doc_chunks: List[DocumentChunk] = []
            chunk_index = 0

            for section in sections:
                content = section["text"].strip()
                if not content:
                    continue

                section_path = list(section["path"])
                page_numbers = list(section["page_numbers"])
                metadata = {
                    "doc_id": doc_id,
                    "source_file": filename,
                    "section_title": section["title"],
                    "section_level": section["level"],
                    "section_path": section_path,
                    "section_path_str": " > ".join(section_path),
                    "page_numbers": page_numbers,
                    "page_span": (
                        [page_numbers[0], page_numbers[-1]]
                        if page_numbers
                        else []
                    ),
                    "contains_table": section["has_tables"],
                    "contains_list": section["has_lists"],
                }


                # Put into a LangChain Document for splitting
                document = self.document_class(
                    page_content=content,
                    metadata=metadata,
                )

                # Split those LangChain Documents into chunks based on text splitter parameters
                split_docs = self.text_splitter.split_documents([document])

                # Process each split document
                for split_doc in split_docs:
                    chunk_text = split_doc.page_content.strip()
                    if not chunk_text:
                        continue
                    chunk_index += 1

                    chunk_metadata = dict(split_doc.metadata)
                    chunk_metadata["chunk_index"] = chunk_index
                    chunk_metadata["chunk_size"] = self.chunk_size
                    chunk_metadata["chunk_overlap"] = self.chunk_overlap
                    chunk_metadata.setdefault("doc_id", doc_id)
                    chunk_metadata.setdefault("source_file", filename)
                    chunk_metadata.setdefault("section_title", section["title"])
                    chunk_metadata.setdefault("section_level", section["level"])
                    chunk_metadata.setdefault("section_path", section_path)
                    chunk_metadata.setdefault(
                        "section_path_str", " > ".join(section_path)
                    )
                    chunk_metadata.setdefault(
                        "page_numbers", page_numbers
                    )
                    chunk_metadata.setdefault(
                        "page_span",
                        [page_numbers[0], page_numbers[-1]]
                        if page_numbers
                        else [],
                    )
                    chunk_metadata.setdefault(
                        "contains_table", section["has_tables"]
                    )
                    chunk_metadata.setdefault(
                        "contains_list", section["has_lists"]
                    )

                    chunk = DocumentChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-chunk-{chunk_index:04d}",
                        content=chunk_text,
                        metadata=chunk_metadata,
                    )
                    doc_chunks.append(chunk)

            chunk_map[doc_id] = doc_chunks

        return chunk_map

    def embed_chunks(
        self,
        chunk_map: Optional[Dict[str, List[DocumentChunk]]] = None,
        batch_size: int = 16,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Generate embeddings for every chunk using a SentenceTransformer model.

        Returns the same mapping but with each `DocumentChunk.embedding` populated.
        """
        for doc_id, chunks in chunk_map.items():
            if not chunks:
                continue

            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
            )

            for chunk, vector in zip(chunks, embeddings):
                chunk.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)

        return chunk_map

    def persist_chunks_to_chromadb(
        self,
        chunk_map: Dict[str, List[DocumentChunk]],
        collection_name: str,
        persist_directory: Path,
        reset_collection: bool = False,
        batch_size: int = 128,
    ) -> None:
        """Write chunk embeddings to a ChromaDB collection.

        A Chroma collection is the logical namespace that groups related vectors,
        documents, and metadata together. Queries are always scoped to a collection,
        so persisting chunks into the same collection keeps the embeddings that fuel
        retrieval for a given dataset in one place.

        Args:
            chunk_map: Mapping of document id -> ordered list of enriched chunks.
            collection_name: Target Chroma collection to upsert into.
            persist_directory: Filesystem location for the Chroma persistent client.
            reset_collection: Drop the existing collection before inserting.
            batch_size: Number of vectors to upsert per request.
        """

        if not chunk_map:
            logger.info("No chunks supplied for ChromaDB persistence.")
            return

        persist_directory = Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        client = PersistentClient(path=str(persist_directory))

        if reset_collection:
            try:
                client.delete_collection(collection_name)
                logger.info("Reset Chroma collection '%s'.", collection_name)
            except ValueError:
                logger.debug(
                    "Collection '%s' did not exist prior to reset.",
                    collection_name,
                )

        collection = client.get_or_create_collection(name=collection_name)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embeddings: List[List[float]] = []

        for doc_id, chunks in chunk_map.items():
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(
                        f"Chunk {chunk.chunk_id} for document {doc_id} has no embedding."
                    )

                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                metadata = self._sanitize_metadata(dict(chunk.metadata))
                metadata.setdefault("doc_id", doc_id)
                metadata["chunk_id"] = chunk.chunk_id
                metadatas.append(metadata)
                embeddings.append(chunk.embedding)

        if not ids:
            logger.info(
                "No chunk content available to persist for collection '%s'.",
                collection_name,
            )
            return

        upsert_fn = getattr(collection, "upsert", None)
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            payload = {
                "ids": ids[start:end],
                "documents": documents[start:end],
                "metadatas": metadatas[start:end],
                "embeddings": embeddings[start:end],
            }
            if upsert_fn is not None:
                upsert_fn(**payload)
            else:
                collection.add(**payload)

        logger.info(
            "Persisted %s chunks to Chroma collection '%s' at %s.",
            len(ids),
            collection_name,
            persist_directory,
        )

    # ------------------------------------------------------------------ Helpers
    def query_collection(
        self,
        query_texts: List[str],
        collection_name: str,
        persist_directory: Path,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return similarity search results for the supplied queries.

        Args:
            query_texts: One or more natural-language queries to embed and search.
            collection_name: Name of the target Chroma collection to read from.
            persist_directory: Filesystem location where the collection is stored.
            top_k: Maximum number of chunks to return per query.
            doc_id: Optional metadata filter to restrict results to a source document.

        Returns:
            Raw Chroma query payload containing the ids, documents, metadata, and
            distances for the closest matches.
        """

        if not query_texts:
            raise ValueError("At least one query string must be provided.")

        persist_directory = Path(persist_directory)
        client = PersistentClient(path=str(persist_directory))

        try:
            collection = client.get_collection(name=collection_name)
        except ValueError as exc:
            raise ValueError(
                f"Collection '{collection_name}' was not found at {persist_directory}."
            ) from exc

        query_embeddings = self.embedder.encode(
            query_texts,
            normalize_embeddings=True,
        )

        metadata_filter: Optional[Dict[str, Any]] = {"doc_id": doc_id} if doc_id else None

        return collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=metadata_filter,
        )
    
    def _ensure_reranker(self) -> CrossEncoder:
        """Lazily load and cache the cross-encoder reranker."""

        if not self.reranker_model_name:
            raise ValueError("Reranker model name not configured.")
        if self._reranker is None:
            self._reranker = CrossEncoder(
                self.reranker_model_name,
                device=self.embedding_device,
                trust_remote_code=True,
                max_length=512,
            )
        return self._reranker

    def rerank_query_results(
        self,
        query: str,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        top_k: int,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float], Optional[List[float]]]:
        """Rerank retrieved chunks using the configured cross-encoder.

        Reranker model is called for each query for each retrieved-to-document pair. (slow)

        Args:
            query: Natural-language query used for retrieval.
            ids: Ordered chunk identifiers returned by the vector search.
            documents: Chunk texts corresponding to each id.
            metadatas: Metadata payloads aligned with `ids`.
            distances: Vector-space similarity distances (cosine or inner-product).
            top_k: Maximum number of reranked results to keep.

        Returns:
            Tuple containing the reranked ids, documents, metadata, and distances,
            along with the list of raw reranker scores.
        """

        if not ids or not self.reranker_model_name:
            return ids, documents, metadatas, distances, None

        reranker = self._ensure_reranker()
        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs)
        indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[: top_k or len(scores)]

        reranked_ids = [ids[idx] for idx in indices]
        reranked_docs = [documents[idx] for idx in indices]
        reranked_metadatas = [metadatas[idx] for idx in indices]
        if distances:
            reranked_distances = [distances[idx] for idx in indices]
        else:
            reranked_distances = [None] * len(indices)
        reranker_scores = [float(scores[idx]) for idx in indices]

        return (
            reranked_ids,
            reranked_docs,
            reranked_metadatas,
            reranked_distances,
            reranker_scores,
        )

    def run_query_loop(
        self,
        collection_name: str,
        persist_directory: Path,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        initial_query: Optional[str] = None,
    ) -> None:
        """Simple REPL to issue similarity queries against the collection.

        Args:
            collection_name: Name of the target Chroma collection to read from.
            persist_directory: Filesystem location where the collection is stored.
            top_k: Maximum number of chunks to return per query.
            doc_id: Optional metadata filter to restrict results to a source document.
            initial_query: Optional query that is executed once before the prompt loop.
        """

        print("\nEnter a query to fetch the top matches (blank line to exit).")
        print("Commands: :doc <doc_id> to filter, :clear to remove filter.")
        active_doc_id = doc_id

        def _execute(query: str) -> None:
            results = self.query_collection(
                query_texts=[query],
                collection_name=collection_name,
                persist_directory=persist_directory,
                top_k=top_k,
                doc_id=active_doc_id,
            )

            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not ids:
                print("No matches found.")
                return

            (
                reranked_ids,
                reranked_docs,
                reranked_metadatas,
                reranked_distances,
                reranker_scores,
            ) = self.rerank_query_results(
                query=query,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                distances=distances,
                top_k=top_k,
            )

            print(f"\nTop {len(reranked_ids)} results:")
            for rank, (chunk_id, distance, doc, metadata, score) in enumerate(
                zip(
                    reranked_ids,
                    reranked_distances,
                    reranked_docs,
                    reranked_metadatas,
                    reranker_scores or [None] * len(reranked_ids),
                ),
                start=1,
            ):
                doc_preview = (doc[:200] + "...") if len(doc) > 200 else doc
                section_path = metadata.get("section_path_str")
                distance_str = f" distance={distance:.4f}" if isinstance(distance, (int, float)) else ""
                score_str = (
                    f" score={score:.4f}" if isinstance(score, (int, float)) else ""
                )
                print(
                    f"[{rank}] chunk={chunk_id}{distance_str}{score_str}"
                    f"\n    section={section_path}"
                    f"\n    text={doc_preview}\n"
                )

        if initial_query:
            _execute(initial_query)

        while True:
            try:
                query = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting query loop.")
                break

            if not query:
                print("Exiting query loop.")
                break

            if query.startswith(":doc"):
                parts = query.split(maxsplit=1)
                if len(parts) == 2 and parts[1]:
                    active_doc_id = parts[1].strip()
                    print(f"Restricting queries to doc_id='{active_doc_id}'.")
                else:
                    print("Usage: :doc <doc_id>")
                continue

            if query == ":clear":
                active_doc_id = None
                print("Cleared doc_id filter.")
                continue

            _execute(query)

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce metadata values into Chroma-compatible scalar types."""

        def _coerce(value: Any) -> Optional[Any]:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple, set)):
                return json.dumps(list(value), ensure_ascii=False)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return {key: _coerce(value) for key, value in metadata.items()}

    def _build_sections(
        self,
        doc: DoclingDocument,
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Traverse the Docling document and build hierarchical sections with
        aggregated text suitable for downstream chunking.

        Depending on what docling tags different things...
        TitleItem: Add # heading at level 1
        SectionHeaderItem: Add heading (## , ###, etc. ) at specified level
        Paragraphs (TextItem): Add as text under current section
        ListItem: Add as bulleted list under current section (- item)
        Tables (TableItem): Convert to markdown table under current section
        Other items: Ignore but track page numbers
        """

        root_title = doc.name or doc_id
        root_section = {
            "title": root_title,
            "level": 0,
            "path": [root_title],
            "parts": [],
            "page_numbers": set(),
            "has_tables": False,
            "has_lists": False,
        }
        sections: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = [root_section]

        for item, _level in doc.iterate_items(with_groups=False, traverse_pictures=True):
            if isinstance(item, SectionHeaderItem):
                level = max(getattr(item, "level", 1), 1)
                title = item.text.strip()
                if not title:
                    continue

                while len(stack) > 1 and stack[-1]["level"] >= level:
                    stack.pop()
                parent = stack[-1] if stack else root_section
                new_section = {
                    "title": title,
                    "level": level,
                    "path": parent["path"] + [title],
                    "parts": [self._format_heading_text(title, level)],
                    "page_numbers": set(),
                    "has_tables": False,
                    "has_lists": False,
                }
                heading_pages = self._extract_page_numbers(item)
                new_section["page_numbers"].update(heading_pages)
                self._update_section_pages(stack, heading_pages)
                sections.append(new_section)
                stack.append(new_section)
                continue

            if isinstance(item, TitleItem):
                title_text = item.text.strip()
                if title_text:
                    heading_line = self._format_heading_text(title_text, level=1)
                    root_section["parts"].append(heading_line)
                    title_pages = self._extract_page_numbers(item)
                    self._update_section_pages(stack, title_pages)
                continue

            target_section = stack[-1] if stack else root_section

            if isinstance(item, TextItem):
                text = item.text.strip()
                if text:
                    target_section["parts"].append(text)
                    self._update_section_pages(stack, self._extract_page_numbers(item))
                continue

            if isinstance(item, ListItem):
                text = item.text.strip()
                if text:
                    bullet = getattr(item, "marker", "-")
                    target_section["parts"].append(f"{bullet} {text}")
                    self._update_section_pages(stack, self._extract_page_numbers(item))
                    target_section["has_lists"] = True
                continue

            # Process Tables by converting it as a markdown.
            if isinstance(item, TableItem):
                table_markdown = self._table_to_markdown(item, doc)
                if table_markdown:
                    target_section["parts"].append(table_markdown)
                    self._update_section_pages(stack, self._extract_page_numbers(item))
                    target_section["has_tables"] = True
                continue

            if isinstance(item, DocItem):
                self._update_section_pages(stack, self._extract_page_numbers(item))

        sections_out: List[Dict[str, Any]] = []
        for section in [root_section, *sections]:
            text = "\n\n".join(section["parts"]).strip()
            if not text:
                continue

            pages_sorted = sorted(section["page_numbers"])
            sections_out.append(
                {
                    "title": section["title"],
                    "level": section["level"],
                    "path": section["path"],
                    "text": text,
                    "page_numbers": pages_sorted,
                    "has_tables": section["has_tables"],
                    "has_lists": section["has_lists"],
                }
            )

        return sections_out

    @staticmethod
    def _format_heading_text(title: str, level: int) -> str:
        prefix = "#" * max(1, min(level, 6))
        return f"{prefix} {title}"

    @staticmethod
    def _extract_page_numbers(item: Any) -> Set[int]:
        pages: Set[int] = set()
        for prov in getattr(item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                pages.add(page_no)
        return pages

    @staticmethod
    def _update_section_pages(
        section_stack: List[Dict[str, Any]], pages: Set[int]
    ) -> None:
        if not pages:
            return
        for section in section_stack:
            section["page_numbers"].update(pages)

    def _table_to_markdown(self, table: TableItem, doc: DoclingDocument) -> str:
        grid = table.data.grid if table.data else []
        if not grid:
            return ""

        rows: List[List[str]] = []
        for row in grid:
            rows.append([getattr(cell, "text", "").strip() for cell in row])

        if not rows:
            return ""

        max_cols = max(len(row) for row in rows)
        normalized_rows = [
            row + [""] * (max_cols - len(row)) for row in rows
        ]

        caption = self._resolve_table_caption(table, doc)
        lines: List[str] = []
        if caption:
            lines.append(f"**Table:** {caption}")

        if max_cols == 0:
            text_rows = [
                " ".join(cell for cell in row if cell).strip()
                for row in normalized_rows
            ]
            text_rows = [row for row in text_rows if row]
            lines.extend(text_rows)
            return "\n".join(lines).strip()

        header = normalized_rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    @staticmethod
    def _resolve_table_caption(table: TableItem, doc: DoclingDocument) -> str:
        captions: List[str] = []
        for ref in getattr(table, "captions", []) or []:
            try:
                caption_item = ref.resolve(doc)
            except Exception:
                continue
            text = getattr(caption_item, "text", "")
            if text:
                captions.append(text.strip())
        return " ".join(captions).strip()

def main() -> None:
    """
    Update the variables below to configure ingestion without touching pipeline internals.
    """
    doc_id = "BrokerComparison"
    initial_question = "What is the definition of earnings for MARKET OPTION 1?"
    collection_name = "hierarchical_chunking"  # Name of the ChromaDB collection
    root = Path(__file__).resolve().parent  # project root
    pipeline = DoclingIngestionPipeline(
        chunk_size=512,
        chunk_overlap=40,
        data_path=root / "data",  # or root / "src" / "data" if that’s where PDFs live
        embedding_model_name="BAAI/bge-large-en-v1.5",
        reranker_model_name="BAAI/bge-reranker-base" # None if no reranking desired
    )

    # # Step 1: Convert PDFs to DoclingDocuments
    # pdfs = pipeline.convert_pdf_documents()

    # Step 2: Build hierarchical chunks from DoclingDocuments
    # # Hierarchical Chunking with Document Metadata
    # chunks = pipeline.build_hierarchical_chunks(doc_map=pdfs)

    # # Step 3: Embed chunks using the configured embedding model
    # chunks_with_embeddings = pipeline.embed_chunks(
    #     chunk_map=chunks,
    #     batch_size=16,
    #     normalize=True,
    #     show_progress_bar=True,
    # )

    # Step 4: Persist chunks to ChromaDB
    # # Add chunks to ChromaDB - can skip if it's already in there
    # pipeline.persist_chunks_to_chromadb(
    #     chunk_map=chunks_with_embeddings,
    #     collection_name=collection_name, # Existing: hierarchical_chunking
    #     persist_directory=root / "chroma_db",
    #     reset_collection=False, # If True, will delete existing collection * NEEDS TO EXIST *
    #     batch_size=256,
    # )

    # We want to Q&A on one document only , so pass in doc_id
    pipeline.run_query_loop(
        collection_name=collection_name,
        persist_directory=root / "chroma_db",
        top_k=5,
        doc_id=doc_id,  # or None to search all documents
        initial_query=initial_question,
    )

if __name__ == "__main__":
    main()
