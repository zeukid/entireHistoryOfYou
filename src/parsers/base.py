"""Shared types and cleaning helpers for all chat-export parsers.

Every parser turns a platform-specific export into a flat list of `Message`
records. `build_dataset.py` then turns those into training examples, so all the
platform-specific mess lives here in the parsers and nowhere else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Message:
    """One message from anyone in the conversation."""

    sender: str
    text: str
    timestamp: str | None = None  # kept as the raw export string; locale-agnostic


# Zero-width Unicode marks WhatsApp/iOS inject into exports: LTR/RTL marks
# (200E/200F), the directional-embedding range (202A-202E), and BOM/ZWNBSP
# (FEFF). These carry no text, so they're deleted. Built from integers with
# chr() so the SOURCE file stays pure ASCII -- literal invisible chars corrupt
# on disk and can't be reviewed. The actual characters only exist at runtime.
_CONTROL_CODEPOINTS = [0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0xFEFF]
_CONTROL_CHARS = re.compile("[" + "".join(map(chr, _CONTROL_CODEPOINTS)) + "]")

# Non-breaking spaces (00A0 regular, 202F narrow). These ARE spaces, so replace
# them with a normal space rather than deleting (deleting would glue words).
_NBSP = re.compile("[" + chr(0x00A0) + chr(0x202F) + "]")

# Optional leading LTR/RTL mark WhatsApp prepends to some system strings.
_BIDI = "[" + chr(0x200E) + chr(0x200F) + "]?"

# Placeholder strings that mean "attachment/system event", not real text.
# Matched case-insensitively against the *whole* (stripped) message body.
_PLACEHOLDER_PATTERNS = [
    r"<media omitted>",
    r"<attached:.*>",
    r".*\b(image|video|audio|gif|sticker|document|contact card) omitted\b.*",
    r"this message was deleted\.?",
    r"you deleted this message\.?",
    _BIDI + r"<this message was edited>",
    r"missed (voice|video) call",
    r"null",
    r"location: https?://\S+",
]
_PLACEHOLDER_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PLACEHOLDER_PATTERNS),
    flags=re.IGNORECASE,
)

# Trailing "<This message was edited>" marker that follows real text.
_EDITED_SUFFIX_RE = re.compile(r"\s*" + _BIDI + r"<this message was edited>\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Strip control marks and the edited-suffix; return '' for placeholders."""
    text = _CONTROL_CHARS.sub("", text)
    text = _NBSP.sub(" ", text)
    text = _EDITED_SUFFIX_RE.sub("", text)
    text = text.strip()
    if not text:
        return ""
    if _PLACEHOLDER_RE.fullmatch(text):
        return ""
    return text
