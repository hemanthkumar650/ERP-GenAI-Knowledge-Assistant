from dataclasses import dataclass
from pathlib import Path
import re

from src.core.config import settings
from src.ingestion.pdf_parser import extract_pdf_text


@dataclass
class DocumentChunk:
    source: str
    source_file: str
    section_id: str
    text: str
    chunk_len: int


NAV_NOISE_PATTERNS = (
    "what's new",
    "related products",
    "try/buy/deploy",
    "get started",
    "train",
    "overview",
    "microsoft dynamics 365 finance documentation",
    "discover how to make the most of dynamics 365",
    "https://learn.microsoft.com",
)

TIMESTAMP_PATTERN = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(am|pm)", re.IGNORECASE)
PAGE_FRACTION_PATTERN = re.compile(r"^\d+\s*/\s*\d+$")


def _is_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    lowered = text.lower()
    if TIMESTAMP_PATTERN.search(lowered):
        return True
    if PAGE_FRACTION_PATTERN.match(lowered):
        return True
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return True
    if any(p in lowered for p in NAV_NOISE_PATTERNS):
        return True
    if len(text) < 3:
        return True
    # Drop fragment lines that are mostly punctuation or isolated menu labels.
    alnum_count = sum(ch.isalnum() for ch in text)
    if alnum_count < 3:
        return True
    if len(text.split()) <= 2 and len(text) < 16:
        return True
    return False


def _clean_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned: list[str] = []
    previous_blank = True
    for line in lines:
        normalized = " ".join(line.strip().split())
        if not normalized:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        if _is_noise_line(normalized):
            continue
        cleaned.append(normalized)
        previous_blank = False
    return "\n".join(cleaned).strip()


def _chunk_paragraphs(paragraphs: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not paragraph:
            continue
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            # Hard split very large paragraphs.
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    current = ""
                    break
                start = max(0, end - overlap_chars)

    if current.strip():
        chunks.append(current.strip())
    return chunks


def load_document_chunks(docs_path: str) -> list[DocumentChunk]:
    base = Path(docs_path)
    chunks: list[DocumentChunk] = []
    patterns = ("*.txt", "*.md", "*.pdf")
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path.suffix.lower() == ".pdf":
                try:
                    content = extract_pdf_text(path)
                except ModuleNotFoundError:
                    continue
            else:
                content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            cleaned = _clean_text(content)
            if not cleaned:
                continue
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            doc_chunks = _chunk_paragraphs(
                paragraphs,
                max_chars=settings.max_chunk_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
            for idx, chunk_text in enumerate(doc_chunks, start=1):
                text = " ".join(chunk_text.strip().split())
                if text:
                    section_id = f"section-{idx}"
                    chunks.append(
                        DocumentChunk(
                            source=f"{path.name}#{section_id}",
                            source_file=path.name,
                            section_id=section_id,
                            text=text,
                            chunk_len=len(text),
                        )
                    )
    return chunks
