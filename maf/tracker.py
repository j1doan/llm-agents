"""Dependency/provenance tracking between stage runs.

Run B depends on run A if B read an artifact A produced (message passing) or, when state tracking is
on, a shared-state key that A wrote. The message-only mode exists as an ablation: it misses the
dependency of the planner on the policy researcher, which flows through shared state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageRun:
    run_id: str
    stage: str
    attempt: int
    is_recovery: bool = False
    status: str = "running"       # running | ok | rejected | timeout | superseded
    read_artifacts: list = field(default_factory=list)   # producer run ids
    read_state: list = field(default_factory=list)       # (key, writer_run_id)
    effect_seqs: list = field(default_factory=list)


class Tracker:
    def __init__(self, order):
        self.order = list(order)
        self.pos = {}                 # stage -> first index in order (O(1) sort key)
        for i, st in enumerate(self.order):
            self.pos.setdefault(st, i)
        self.runs: dict = {}
        self.artifacts: dict = {}     # run_id -> content
        self.current: dict = {}       # stage -> run_id of the live run
        self.time_s = 0.0             # bookkeeping overhead

    def note_artifact_read(self, run: StageRun, producer_run_id: str):
        t0 = time.perf_counter()
        run.read_artifacts.append(producer_run_id)
        self.time_s += time.perf_counter() - t0

    def note_state_read(self, run: StageRun, key: str, writer):
        t0 = time.perf_counter()
        run.read_state.append((key, writer))
        self.time_s += time.perf_counter() - t0

    def downstream(self, run_id: str, track_state: bool) -> list:
        """Run ids of run_id plus every live run that transitively depends on it, in stage order."""
        t0 = time.perf_counter()
        closure = {run_id}
        live = [self.runs[r] for r in self.current.values()]
        changed = True
        while changed:
            changed = False
            for r in live:
                if r.run_id in closure:
                    continue
                via_msg = any(a in closure for a in r.read_artifacts)
                via_state = track_state and any(w in closure for _, w in r.read_state)
                if via_msg or via_state:
                    closure.add(r.run_id)
                    changed = True
        ordered = sorted(closure, key=lambda rid: self.pos[self.runs[rid].stage])
        self.time_s += time.perf_counter() - t0
        return ordered
