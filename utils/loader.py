from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


NOISE_PATTERNS = (
    "confidential",
    "page ",
    "copyright",
    "all rights reserved",
    "table of contents",
    "intentionally left blank",
    "revision history",
    "employee handbook",
    "policy manual",
)

TIMESTAMP_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(am|pm)", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")


def _is_noise_line(line: str) -> bool:
    text = " ".join(line.strip().split())
    if not text:
        return True
    lowered = text.lower()
    if TIMESTAMP_RE.search(lowered):
        return True
    if URL_RE.search(lowered):
        return True
    if PAGE_NUMBER_RE.match(lowered):
        return True
    if len(text) <= 2:
        return True
    if len(text.split()) == 1 and len(text) < 6:
        return True
    if re.fullmatch(r"[A-Z0-9\-\._/ ]{1,40}", text) and len(text.split()) <= 3:
        # Header-like artifacts e.g., "HR-001", "APPROVED", "PAGE 5".
        return True
    if any(p in lowered for p in NOISE_PATTERNS):
        return True
    return False


def _clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned: list[str] = []
    prev_blank = True
    for line in lines:
        norm = " ".join(line.strip().split())
        if _is_noise_line(norm):
            if not norm:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            continue
        cleaned.append(norm)
        prev_blank = False
    return "\n".join(cleaned).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    cleaned = _clean_text("\n\n".join(pages))
    # Normalize excessive blank lines but preserve paragraph breaks.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_policy_documents(data_dir: str) -> list[dict]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    for pdf_path in sorted(root.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        if not text:
            continue
        documents.append(
            {
                "text": text,
                "source": pdf_path.name,
            }
        )
    return documents
