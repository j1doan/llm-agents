"""Small helpers shared across the package."""
from __future__ import annotations

import json
import re
import zlib


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def est_tokens(text: str) -> int:
    """Rough token estimate, only used by the mock backend."""
    return max(1, len(text) // 4)


def extract_json(text):
    """Return the first JSON object found in text, or None."""
    if text is None:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None
