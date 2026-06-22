"""Parse an exported WhatsApp `_chat.txt` into a list of `Message`.

WhatsApp's "Export chat" produces a plain-text file in one of two broad
layouts depending on the phone OS / locale:

    iOS:      [15/01/2023, 9:42:13 PM] Alice: Hey, you around?
    Android:  15/01/2023, 21:42 - Alice: Hey, you around?

A single message can span multiple lines (the body just continues on the next
line with no timestamp). System lines ("Messages and calls are end-to-end
encrypted", "Alice created group ...") share the timestamp prefix but have no
"Sender: " part, so we detect and drop them.
"""

from __future__ import annotations

from pathlib import Path

import re

from .base import Message, clean_text

# A line that *starts a new message*. We first detect the timestamp prefix,
# then try to split off "Sender: body". Two prefix flavours:
#   iOS:     [ ... ]  (square brackets)
#   Android: ... -    (trailing dash separator)
_IOS_PREFIX = re.compile(r"^\[(?P<date>[^\]]+?)\]\s*(?P<rest>.*)$", re.DOTALL)
_ANDROID_PREFIX = re.compile(
    r"^(?P<date>\d{1,2}[/.]\d{1,2}[/.]\d{2,4},?\s+"
    r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?)\s-\s(?P<rest>.*)$",
    re.DOTALL,
)
# Splits "Sender Name: message body" -- sender names never contain a colon.
_SENDER_SPLIT = re.compile(r"^(?P<sender>[^:]{1,80}?):\s(?P<text>.*)$", re.DOTALL)


def _split_prefix(line: str) -> tuple[str, str] | None:
    """Return (date, rest) if `line` starts a new message, else None."""
    for pat in (_IOS_PREFIX, _ANDROID_PREFIX):
        m = pat.match(line)
        if m:
            return m.group("date"), m.group("rest")
    return None


def parse(path: str | Path) -> list[Message]:
    """Parse a WhatsApp export file into chronological `Message` records.

    Placeholder/system content is dropped here; multi-line messages are
    stitched back together. Order is preserved as in the export.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    messages: list[Message] = []
    # Accumulator for the message currently being built (handles multi-line).
    cur_sender: str | None = None
    cur_date: str | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_sender, cur_date, cur_lines
        if cur_sender is not None:
            text = clean_text("\n".join(cur_lines))
            if text:
                messages.append(Message(sender=cur_sender, text=text, timestamp=cur_date))
        cur_sender, cur_date, cur_lines = None, None, []

    for line in raw.splitlines():
        prefix = _split_prefix(line)
        if prefix is None:
            # Continuation of the current message's body.
            if cur_sender is not None:
                cur_lines.append(line)
            continue

        date, rest = prefix
        flush()  # the previous message is complete
        sender_match = _SENDER_SPLIT.match(rest)
        if sender_match is None:
            # Timestamped but no "Sender:" -> system event. Skip it.
            continue
        cur_sender = sender_match.group("sender").strip()
        cur_date = date.strip()
        cur_lines = [sender_match.group("text")]

    flush()
    return messages
