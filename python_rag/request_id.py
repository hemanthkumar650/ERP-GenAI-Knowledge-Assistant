from __future__ import annotations

import re
import uuid

_MAX_LEN = 128
_SAFE_INCOMING = re.compile(r"^[a-zA-Z0-9._-]+$")


def normalize_incoming(raw: str | None) -> str | None:
    if raw is None:
        return None
    trimmed = raw.strip()
    if not trimmed or len(trimmed) > _MAX_LEN or _SAFE_INCOMING.fullmatch(trimmed) is None:
        return None
    return trimmed


def assign_request_id(header_value: str | None) -> str:
    return normalize_incoming(header_value) or str(uuid.uuid4())
