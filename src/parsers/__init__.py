"""Parser registry. Add a new platform by dropping in a module with a
`parse(path) -> list[Message]` function and registering it here.
"""

from __future__ import annotations

from .base import Message
from . import whatsapp

# name -> parse function. Phase 7 adds: imessage, telegram, discord.
PARSERS = {
    "whatsapp": whatsapp.parse,
}


def parse(platform: str, path: str) -> list[Message]:
    if platform not in PARSERS:
        raise ValueError(
            f"unknown platform {platform!r}; available: {', '.join(sorted(PARSERS))}"
        )
    return PARSERS[platform](path)


__all__ = ["Message", "PARSERS", "parse"]
