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
EFFECTIVE_DATE_RE = re.compile(
    r"\b(?:effective(?:\s+date)?[:\s]+)([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)
POLICY_TYPE_RE = re.compile(r"\b(?:type of policy|policy type)[:\s]+([A-Za-z0-9&/\- ]{2,80})", re.IGNORECASE)
DEPARTMENT_RE = re.compile(
    r"\b(?:sponsoring department\(s\)|department)[:\s]+([A-Za-z0-9&/\- ,]{2,120})",
    re.IGNORECASE,
)
VERSION_RE = re.compile(r"\b(?:version|rev(?:ision)?)[:\s#]+([A-Za-z0-9.\-]{1,30})", re.IGNORECASE)
YEAR_IN_NAME_RE = re.compile(r"(19|20)\d{2}")


def _is_noise_line(line: str) -> bool:
    text = " ".join(line.strip().split())
    if not text:
        return True
    lowered = text.lower()
    if TIMESTAMP_RE.search(lowered) or URL_RE.search(lowered) or PAGE_NUMBER_RE.match(lowered):
        return True
    if len(text) <= 2:
        return True
    if len(text.split()) == 1 and len(text) < 6:
        return True
    if re.fullmatch(r"[A-Z0-9\-\._/ ]{1,40}", text) and len(text.split()) <= 3:
        return True
    if any(pattern in lowered for pattern in NOISE_PATTERNS):
        return True
    return False


def _clean_text(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned: list[str] = []
    prev_blank = True
    for line in lines:
        normalized = " ".join(line.strip().split())
        if _is_noise_line(normalized):
            if not normalized:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            continue
        cleaned.append(normalized)
        prev_blank = False
    return "\n".join(cleaned).strip()


def _normalize_meta_value(value: str, max_len: int = 80) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        return "unknown"
    return text[:max_len]


def _extract_policy_metadata(cleaned_text: str, source_name: str) -> dict:
    header = "\n".join((cleaned_text or "").splitlines()[:40])
    policy_type_match = POLICY_TYPE_RE.search(header)
    effective_date_match = EFFECTIVE_DATE_RE.search(header)
    department_match = DEPARTMENT_RE.search(header)
    version_match = VERSION_RE.search(header)
    year_match = YEAR_IN_NAME_RE.search(source_name or "")

    metadata = {
        "policy_type": _normalize_meta_value(policy_type_match.group(1) if policy_type_match else "unknown"),
        "effective_date": _normalize_meta_value(effective_date_match.group(1) if effective_date_match else "unknown"),
        "department": _normalize_meta_value(department_match.group(1) if department_match else "unknown"),
        "version": _normalize_meta_value(
            version_match.group(1) if version_match else (year_match.group(0) if year_match else "unknown")
        ),
    }
    return metadata


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    cleaned = _clean_text("\n\n".join(pages))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_policy_documents(data_dir: str) -> list[dict]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    for pdf_path in sorted(root.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        if not text:
            continue
        metadata = _extract_policy_metadata(text, pdf_path.name)
        documents.append({"text": text, "source": pdf_path.name, "metadata": metadata})
    return documents
