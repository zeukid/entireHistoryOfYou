"""Turn a chat export into chat-format training JSONL for QLoRA fine-tuning.

Pipeline:
    export file --[parser]--> Messages
                --[merge consecutive same-sender]--> turns
                --[sliding window ending on a YOUR-reply]--> examples
                --[PII scrub + role labels]--> train.jsonl

Each output line is one example:
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
where YOUR messages are the `assistant` (what the model learns to imitate) and
everyone else is the `user`.

Usage:
    # 1. See who's in the chat (so you know what to pass as --me):
    python -m src.build_dataset --export exports/_chat.txt --list-senders

    # 2. Build the dataset:
    python -m src.build_dataset --export exports/_chat.txt --me "Your Name" \
        --out data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .parsers import Message, parse

# ---- PII scrubbing -------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone numbers: optional +, then 7+ digits possibly broken by spaces/-/().
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")


def scrub(text: str) -> str:
    """Redact emails and phone numbers. Conservative: keeps everything else."""
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return text


# ---- Turn building -------------------------------------------------------


@dataclass
class Turn:
    sender: str
    text: str


def merge_consecutive(messages: list[Message]) -> list[Turn]:
    """Collapse runs of messages from the same sender into one turn.

    People fire off several messages in a row; to the model that's a single
    conversational turn, so we join them with newlines.
    """
    turns: list[Turn] = []
    for m in messages:
        if turns and turns[-1].sender == m.sender:
            turns[-1].text += "\n" + m.text
        else:
            turns.append(Turn(sender=m.sender, text=m.text))
    return turns


def build_examples(
    turns: list[Turn],
    me: str,
    ctx_turns: int = 6,
    min_chars: int = 2,
) -> list[dict]:
    """Sliding window: every YOUR-turn becomes a target, with preceding context.

    The context window is collapsed into alternating user/assistant roles so it
    is valid for any chat template (consecutive same-role turns are joined).
    """
    examples: list[dict] = []
    for i, turn in enumerate(turns):
        if turn.sender != me:
            continue
        target = scrub(turn.text).strip()
        if len(target) < min_chars:
            continue
        window = turns[max(0, i - ctx_turns):i]
        # Need at least one *other-person* turn before the reply, otherwise
        # the example is "you talking to yourself" with no prompt.
        if not any(t.sender != me for t in window):
            continue

        msgs: list[dict] = []
        for t in window:
            role = "assistant" if t.sender == me else "user"
            content = scrub(t.text).strip()
            if not content:
                continue
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n" + content  # keep roles alternating
            else:
                msgs.append({"role": role, "content": content})
        # The example must end on a `user` turn so the target is the reply to it.
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        if not msgs:
            continue
        msgs.append({"role": "assistant", "content": target})
        examples.append({"messages": msgs})
    return examples


def dedup_and_cap(examples: list[dict], cap: int) -> tuple[list[dict], int, int]:
    """Drop exact-duplicate examples, and cap how often any one reply repeats.

    Chat data is full of identical filler ("ok", "lol", "haha"). Left unchecked
    the model just learns to spam those. We keep at most `cap` examples whose
    *target reply* is identical, and drop fully duplicate (context+reply) ones.
    """
    seen: set[str] = set()
    reply_counts: Counter = Counter()
    out, dropped_dupe, dropped_cap = [], 0, 0
    for ex in examples:
        key = json.dumps(ex["messages"], ensure_ascii=False, sort_keys=True)
        if key in seen:
            dropped_dupe += 1
            continue
        seen.add(key)
        target = ex["messages"][-1]["content"]
        reply_counts[target] += 1
        if reply_counts[target] > cap:
            dropped_cap += 1
            continue
        out.append(ex)
    return out, dropped_dupe, dropped_cap


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- CLI -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="path to the chat export file")
    ap.add_argument("--platform", default="whatsapp", help="export platform (default: whatsapp)")
    ap.add_argument("--me", help="the sender name to learn (your name as it appears in the chat)")
    ap.add_argument("--out", default="data/train.jsonl", help="output JSONL path")
    ap.add_argument("--ctx-turns", type=int, default=6, help="context turns before each reply")
    ap.add_argument("--min-chars", type=int, default=2, help="drop your replies shorter than this")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction held out for eval (writes val.jsonl alongside --out)")
    ap.add_argument("--cap-identical-replies", type=int, default=20,
                    help="max examples sharing an identical target reply")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for the split")
    ap.add_argument("--list-senders", action="store_true",
                    help="just print who's in the chat and exit")
    args = ap.parse_args()

    messages = parse(args.platform, args.export)
    if not messages:
        raise SystemExit("No messages parsed -- is the export format correct?")

    counts = Counter(m.sender for m in messages)
    if args.list_senders:
        print(f"{len(messages)} messages from {len(counts)} senders:\n")
        for name, n in counts.most_common():
            print(f"  {n:>7,}  {name}")
        return

    if not args.me:
        raise SystemExit(
            "Pass --me with your name. Run with --list-senders to see the options."
        )
    if args.me not in counts:
        raise SystemExit(
            f"{args.me!r} not found among senders. Run --list-senders to see exact names."
        )

    turns = merge_consecutive(messages)
    examples = build_examples(turns, args.me, args.ctx_turns, args.min_chars)
    if not examples:
        raise SystemExit("No training examples produced -- try a smaller --min-chars.")

    examples, dropped_dupe, dropped_cap = dedup_and_cap(examples, args.cap_identical_replies)
    random.Random(args.seed).shuffle(examples)
    n_val = int(len(examples) * args.val_frac)
    val, train = examples[:n_val], examples[n_val:]

    out = Path(args.out)
    write_jsonl(out, train)
    print(f"Wrote {len(train):,} train examples to {out}")
    if val:
        val_path = out.parent / "val.jsonl"
        write_jsonl(val_path, val)
        print(f"Wrote {len(val):,} val examples to {val_path}")
    print(
        f"  (deduped {dropped_dupe:,}, capped {dropped_cap:,}; "
        f"from {counts[args.me]:,} of your messages across {len(messages):,} total)"
    )


if __name__ == "__main__":
    main()
