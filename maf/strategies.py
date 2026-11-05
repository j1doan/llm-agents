"""Recovery strategies. All receive the same fault schedule and the same inference budget ceilings.

Per the project spec, `full_restart`, `selective` and `selective_msg` are told the identity of the faulty
run (so recovery quality is measured separately from detection). `verify_all` has to detect faults itself
with an extra verifier LLM call after every stage -- that cost is the point of the comparison.

  none           no recovery; measures raw propagation
  full_restart   when the fault is identified, undo everything reversible and rerun every stage
  verify_all     verify each stage's output before committing it; rerun the stage if rejected
  selective      dependency-aware invalidation (message AND shared-state dependencies)
  selective_msg  ablation: dependency tracking from messages only (misses shared-state reads)
  verify_hybrid  verify_all with a deterministic rule check (maf/rules.py) before the LLM verifier. A rule
                 rejection skips the LLM call; if the last attempt still breaks the rules, the stage is committed
                 with the violating actions or fields removed (D28).
  full_restart_gated, selective_gated
                 twins of full_restart / selective that hold emails and transfers in an outbox (D20). At the end
                 of recovery only effects of runs outside the invalidation closure are released; emails are
                 idempotent on (to, subject, body).
"""
from __future__ import annotations

import logging

from .workflow import STAGES, Episode, StageTimeout

log = logging.getLogger(__name__)


def run_none(ep: Episode):
    log.info("strategy none")
    try:
        for stage in STAGES:
            ep.execute_stage(stage)
    except StageTimeout:
        ep.aborted = True
        log.warning("none: aborted on timeout")
    ep.mark_first_pass_done()


def run_full_restart(ep: Episode):
    log.info("strategy full_restart")
    try:
        for stage in STAGES:
            ep.execute_stage(stage)
    except StageTimeout:
        ep.aborted = True
    ep.mark_first_pass_done()
    first_pass = list(ep.tracker.runs) if ep.fault_fired else []
    if ep.fault_fired:
        log.info("full_restart: rolling back everything and rerunning all stages")
        ep.rollback(first_pass)
        ep.aborted = False
        ep.in_recovery = True
        for stage in STAGES:
            ep.execute_stage(stage)
        ep.in_recovery = False
    ep.release_outbox(first_pass)


def _run_selective(ep: Episode, track_state: bool):
    log.info("strategy selective track_state=%s", track_state)
    for stage in STAGES:
        try:
            ep.execute_stage(stage)
        except StageTimeout:
            log.info("selective: retrying %s after timeout", stage)
            ep.in_recovery = True       # nothing downstream has run yet: retry just this stage
            ep.execute_stage(stage)
            ep.in_recovery = False
    ep.mark_first_pass_done()
    closure = []
    if ep.fault_fired and ep.fault.kind != "timeout":
        closure = ep.tracker.downstream(ep.fault_run_id, track_state)
        stages = {ep.tracker.runs[r].stage for r in closure}
        log.info("selective: fault in %s -> invalidating %s (runs %s)", ep.fault_run_id, sorted(stages),
                 sorted(closure))
        ep.rollback(closure)
        ep.in_recovery = True
        for stage in STAGES:
            if stage in stages:
                ep.execute_stage(stage)
        ep.in_recovery = False
    ep.release_outbox(closure)


def run_selective(ep: Episode):
    _run_selective(ep, track_state=True)


def run_selective_msg(ep: Episode):
    _run_selective(ep, track_state=False)


def _run_verify(ep: Episode, rules: bool):
    n = ep.cfg.max_verify_attempts
    for stage in STAGES:
        for attempt in range(n):
            last = attempt == n - 1
            try:
                if ep.execute_stage(stage, verify=True, force_commit=last, rules=rules) is not None:
                    break
            except StageTimeout:
                log.warning("verify: %s timed out (attempt %d/%d)", stage, attempt + 1, n)
            ep.in_recovery = True       # any further attempt for this stage is a recovery run
        ep.in_recovery = False
    ep.mark_first_pass_done()


def run_verify_all(ep: Episode):
    log.info("strategy verify_all")
    _run_verify(ep, rules=False)


def run_verify_hybrid(ep: Episode):
    log.info("strategy verify_hybrid")
    _run_verify(ep, rules=True)


def run_full_restart_gated(ep: Episode):
    ep.gated = True
    run_full_restart(ep)


def run_selective_gated(ep: Episode):
    ep.gated = True
    _run_selective(ep, track_state=True)


STRATEGIES = {
    "none": run_none,
    "full_restart": run_full_restart,
    "verify_all": run_verify_all,
    "selective": run_selective,
    "selective_msg": run_selective_msg,
    "full_restart_gated": run_full_restart_gated,
    "selective_gated": run_selective_gated,
    "verify_hybrid": run_verify_hybrid,
}
