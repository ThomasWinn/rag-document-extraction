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
from typing import Any, Dict, List, Optional, Set

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


logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_path = Path(data_path)
        self.use_cache = use_cache

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
        documents: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[DocumentChunk]]:
        """
        Convert PDF documents (if not provided) and generate LangChain-ready chunks.

        If `documents` is supplied, it should map filenames to `DoclingDocument`
        instances (or full `ConversionResult` objects). The chunks retain section
        hierarchy metadata such as section title, path, and page numbers to support
        structured retrieval workflows.
        """

        doc_map = documents or self.convert_pdf_documents()
        chunk_map: Dict[str, List[DocumentChunk]] = {}

        for filename, value in doc_map.items():
            doc = value.document if hasattr(value, "document") else value
            if not isinstance(doc, DoclingDocument):
                logger.warning("Unsupported document payload for %s; skipping", filename)
                continue

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

                document = self.document_class(
                    page_content=content,
                    metadata=metadata,
                )
                split_docs = self.text_splitter.split_documents([document])

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

    # ------------------------------------------------------------------ Helpers
    def _build_sections(
        self,
        doc: DoclingDocument,
        doc_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Traverse the Docling document and build hierarchical sections with
        aggregated text suitable for downstream chunking.
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

    # ------------------------------------------------------------------ Config
    input_path = Path("data")
    chunk_size = 512
    chunk_overlap = 40

    # ------------------------------------------------------------------ Run
    pipeline = DoclingIngestionPipeline(
        chunk_size,
        chunk_overlap,
        data_path=input_path,
    )
    pdfs = pipeline.convert_pdf_documents()
    chunks = pipeline.build_hierarchical_chunks(documents=pdfs)

    # Process chunks as needed
    for chunk in chunks:
        print(chunk)

if __name__ == "__main__":
    main()
