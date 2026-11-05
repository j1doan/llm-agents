"""Per-event trace (D6, D22): one JSON line per event, flushed as it happens, so a crash keeps the trace.

Record: id, episode (model|seed|fault|strategy), attempt, stage, run_id, kind, parents, t, payload.
kinds: message artifact fault state_read state_write tool_action verify rollback episode_end.
t = simulated seconds (LLM latency + charged timeouts). An attempt is complete only if it has an episode_end.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

_LOCK = threading.Lock()    # episodes may run in parallel threads and share one file


def load_attempts(path) -> dict:
    """episode key -> highest attempt number already in the events file (a crash leaves events but no result
    record, so the results file alone cannot say how many attempts happened)."""
    best = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                best[rec["episode"]] = max(best.get(rec["episode"], 0), rec["attempt"])
    except FileNotFoundError:
        pass
    return best


class EventTrace:
    def __init__(self, path, episode, attempt):
        self.episode, self.attempt, self.n = episode, attempt, 0
        self.time_s = 0.0           # time spent building and writing events
        self._f = open(path, "a", encoding="utf-8")

    def emit(self, kind, stage, run_id, parents, t, payload, prompt=None) -> str:
        t0 = time.perf_counter()
        self.n += 1
        eid = f"e{self.n}"
        if prompt is not None:
            payload = dict(payload, prompt_sha256=hashlib.sha256(
                json.dumps(prompt, sort_keys=True).encode("utf-8")).hexdigest())
        line = json.dumps({"id": eid, "episode": self.episode, "attempt": self.attempt, "stage": stage,
                           "run_id": run_id, "kind": kind, "parents": list(parents), "t": round(t, 6),
                           "payload": payload}, default=str) + "\n"
        with _LOCK:
            self._f.write(line)
            self._f.flush()
        self.time_s += time.perf_counter() - t0
        return eid

    def close(self):
        self._f.close()
