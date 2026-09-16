"""RWKV trie tokenizer (rwkv_vocab_v20230424.txt).

The vocab file has one line per token:

    <id> <repr> <byte_len>

e.g. ``1 '\\x00' 1`` — token id 1 is the single byte 0x00 (BOS ``<s>``).
Token id 0 is EOS (empty). Encoding is greedy longest-match over the byte
string, which is exactly how the official RWKV tokenizer works.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ["RWKVTrieTokenizer"]


def _parse_line(line: str) -> tuple[int, bytes] | None:
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(\d+)\s+(.*)$", line)
    if not m:
        return None
    tid = int(m.group(1))
    rest = m.group(2)
    # rest: "<repr> <byte_len>"  (repr may contain escaped chars, no spaces)
    parts = rest.rsplit(" ", 1)
    if len(parts) != 2:
        return None
    tok_repr, byte_len = parts[0], int(parts[1])
    tok = _decode_repr(tok_repr)
    if tok is None or len(tok) != byte_len:
        return None
    return tid, tok


def _decode_repr(repr_str: str) -> bytes | None:
    """Decode the repr like ``b'\\x00abc'`` (handles \\x, \\n, \\t, \\\\, ...)."""
    s = repr_str
    if s.startswith(("b'", "b\"")):
        s = s[2:-1]
    elif s.startswith(("'", '"')):
        s = s[1:-1]
    else:
        return None
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if i + 1 >= len(s):
                return None
            n = s[i + 1]
            if n == "x":
                if i + 3 >= len(s):
                    return None
                try:
                    out.append(int(s[i + 2:i + 4], 16))
                except ValueError:
                    return None
                i += 4
                continue
            mapping = {"n": 0x0A, "t": 0x09, "r": 0x0D, "\\": 0x5C, "'": 0x27, '"': 0x22}
            if n in mapping:
                out.append(mapping[n])
                i += 2
                continue
            return None
        else:
            b = c.encode("utf-8")
            out.extend(b)
            i += 1
    return bytes(out)


class RWKVTrieTokenizer:
    def __init__(self, vocab_path: str | Path):
        self.root: dict[bytes, object] = {}
        self.tokens: dict[int, bytes] = {0: b""}  # id 0 = EOS
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                tid, tok = parsed
                self.tokens[tid] = tok
                node = self.root
                for ch in tok:
                    b = bytes([ch])
                    node = node.setdefault(b, {})
                node["id"] = tid

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        ids: list[int] = []
        i = 0
        n = len(data)
        while i < n:
            node = self.root
            best = -1
            j = i
            while j < n:
                b = bytes([data[j]])
                node = node.get(b)
                if node is None:
                    break
                if "id" in node:
                    best = node["id"]
                j += 1
            if best < 0:
                # fallback: single byte (should not happen with the full vocab)
                best = 1 + data[i]
            ids.append(best)
            i = j if j > i else i + 1
        return ids

    def decode(self, ids: list[int]) -> str:
        out = bytearray()
        for tid in ids:
            out.extend(self.tokens.get(tid, b"\ufffd"))
        return out.decode("utf-8", errors="replace")
