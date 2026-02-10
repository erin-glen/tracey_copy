"""Deterministic helpers for parsing/decoding CodeAct payloads."""

from __future__ import annotations

import base64
import re
from typing import Any


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_LONG_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


def find_codeact_parts(output_obj: Any) -> list[dict]:
    """Find codeact parts at top-level or nested payload positions."""
    if not isinstance(output_obj, (dict, list)):
        return []

    found: list[dict] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if "codeact_parts" in obj and isinstance(obj["codeact_parts"], list):
                for part in obj["codeact_parts"]:
                    if isinstance(part, dict):
                        found.append(part)
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _walk(item)

    _walk(output_obj)
    return found


def decode_maybe_base64(content: Any) -> tuple[str, bool, int]:
    """Conservatively decode base64 payloads when input looks trustworthy."""
    raw = str(content or "")
    candidate = raw.strip()
    if len(candidate) < 16 or not _BASE64_RE.fullmatch(candidate):
        return raw, False, 0

    try:
        decoded = base64.b64decode(candidate, validate=False)
        text = decoded.decode("utf-8", errors="replace")
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
        ratio = printable / max(1, len(text))
        if ratio >= 0.85:
            return text, True, 0
        return raw, False, 0
    except Exception:
        return raw, False, 1


def iter_decoded_codeact_parts(output_obj: Any) -> list[dict]:
    """Return normalized decoded parts for deterministic downstream processing."""
    normalized: list[dict] = []
    for part in find_codeact_parts(output_obj):
        ptype = str(part.get("type") or "").strip().lower()
        content = part.get("content")
        decoded, decoded_from_b64, decode_error = decode_maybe_base64(content)
        normalized.append(
            {
                "type": ptype,
                "decoded": decoded,
                "decoded_from_b64": decoded_from_b64,
                "decode_error": int(decode_error),
            }
        )
    return normalized


def redact_secrets(text: str) -> str:
    """Redact common token-like patterns from code snippets."""
    out = str(text or "")
    out = re.sub(r"(?im)(authorization\s*:\s*bearer\s+)[^\s\"']+", r"\1<REDACTED>", out)
    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*", r"\1<REDACTED>", out)
    out = re.sub(r"(?i)\b(api[_-]?key|apikey|token)\b\s*[:=]\s*([\"'])?[^\s,\}\]\"']+\2?", r"\1=<REDACTED>", out)
    out = _LONG_BLOB_RE.sub("<REDACTED_BLOB>", out)
    return out


def truncate_text(text: str, max_chars: int) -> str:
    """Deterministically truncate text with ellipsis."""
    txt = str(text or "")
    lim = max(0, int(max_chars))
    if lim == 0:
        return ""
    if len(txt) <= lim:
        return txt
    if lim <= 3:
        return "." * lim
    return txt[: lim - 3] + "..."
