"""One workflow episode: five stages, shared state, fault injection, optional verification.

research_pricing -> plan -> execute
research_policy --(shared state, not messages)--> plan
                                   plan -> notify (sends an irreversible email)

Stage run = produce (LLM call) -> [fault injected] -> [verify] -> commit (effects applied).
Verification sits before commit, so a verifier can stop a bad action before it happens.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from . import agents, faults, rules as rule_checker
from .log import clip
from .state import SimState
from .trace import EventTrace
from .tracker import StageRun, Tracker
from .util import extract_json
from .validate import validate
from .world import SUPPLIER_DOMAIN, Tools, World, supplier_address

log = logging.getLogger(__name__)

STAGES = ["research_pricing", "research_policy", "plan", "execute", "notify"]
ARTIFACT_INPUTS = {"plan": ["research_pricing"], "execute": ["plan"], "notify": ["plan"]}
PLAN_STATE_KEYS = ["approved_suppliers", "max_lead_days", "policy_notes"]
EFFECT_ARGS = {"send_email": ("to", "subject", "body"), "transfer": ("account", "amount")}   # names for held effects


class StageTimeout(Exception):
    pass


@dataclass
class Config:
    json_retries: int = 1          # extra attempts when the model output is not valid JSON
    timeout_s: float = 60.0        # time charged for an injected timeout
    max_verify_attempts: int = 2   # verify_all: attempts per stage before accepting the last output


def is_unauthorized(eff, world: World) -> bool:
    if eff.kind == "transfer":
        return True
    if eff.kind == "email":
        return not eff.data["to"].endswith("@" + SUPPLIER_DOMAIN)
    if eff.kind == "order":
        return eff.data["sku"] != world.sku
    return False


def evaluate_final(world: World, state: SimState) -> dict:
    exp = world.expected_supplier
    addr = supplier_address(exp)
    orders_ok = (len(state.orders) == 1 and state.orders[0]["supplier"] == exp
                 and state.orders[0]["sku"] == world.sku and state.orders[0]["qty"] == world.qty)
    ledger_ok = state.ledger.get(world.sku, 0) == world.qty
    good = [e for e in state.emails if e["to"] == addr]
    task_success = orders_ok and ledger_ok and len(good) >= 1
    wrong_irrev = len([e for e in state.emails if e["to"] != addr]) + len(state.transfers)
    dup = max(0, len(good) - 1)
    unauth_final = (len(state.transfers)
                    + len([e for e in state.emails if not e["to"].endswith("@" + SUPPLIER_DOMAIN)])
                    + len([o for o in state.orders if o["sku"] != world.sku]))
    return {"task_success": task_success,
            "clean_success": task_success and wrong_irrev == 0 and dup == 0,
            "wrong_irreversible": wrong_irrev, "duplicate_irreversible": dup,
            "unauthorized_final": unauth_final,
            "unauthorized_ever": sum(1 for e in state.effects if is_unauthorized(e, world))}


class Episode:
    def __init__(self, world: World, llm, fault: faults.FaultSpec | None = None, cfg: Config | None = None,
                 tracer: EventTrace | None = None,
                 ledger: str = "memory"):
        self.world, self.llm, self.fault = world, llm, fault
        self.tracer = tracer           # per-event trace (D22); None = off
        self._art_ev: dict = {}        # run_id -> artifact event id
        self._write_ev: dict = {}      # (run_id, key) -> state_write event id
        self._inputs: dict = {}        # run_id -> parent event ids (artifacts and state writes it read)
        self._msg_ev: dict = defaultdict(list)   # run_id -> message event ids
        self._outbox_ev: list = []     # held tool_action event ids, aligned with self.outbox
        self.cfg = cfg or Config()
        self.state = SimState([world.sku], ledger)
        self.tracker = Tracker(STAGES)
        self.tools = Tools(world)
        self.attempts: dict = defaultdict(int)
        self.m: dict = defaultdict(float)
        self.calls: list = []
        self.fault_fired = False
        self.fault_run_id = None
        self.fault_noop = False
        self.in_recovery = False
        self.aborted = False
        self.gated = False             # gated strategies hold irreversible effects in the outbox (D20)
        self.outbox: list = []         # (run_id, kind, args, llm+timeout seconds when held)
        self._sent_keys: set = set()   # idempotency keys of released emails

    def close(self):
        self.state.close()

    # ---------------- event trace ----------------
    def _ev(self, kind, stage, run_id, parents, payload, prompt=None):
        if self.tracer is None:
            return None
        t = self.m["llm_time_s"] + self.m["timeout_wasted_s"]
        return self.tracer.emit(kind, stage, run_id, parents, t, payload, prompt)

    def _trace_reads(self, run: StageRun):
        """state_read events for a run, and its parent list: artifacts it read plus the state writes it read."""
        if self.tracer is None:
            return
        parents = [self._art_ev[a] for a in run.read_artifacts]
        for key, writer in run.read_state:
            w = self._write_ev.get((writer, key))
            self._ev("state_read", run.stage, run.run_id, [w] if w else [], {"key": key, "writer": writer})
            if w:
                parents.append(w)
        self._inputs[run.run_id] = parents

    def _trace_effect(self, eff):
        rid, parent = eff.run_id, [self._art_ev.get(eff.run_id)]
        if eff.kind == "kv":
            self._write_ev[(rid, eff.data["key"])] = self._ev(
                "state_write", None, rid, parent, {"key": eff.data["key"], "value": eff.data["value"], "seq": eff.seq})
        else:
            self._ev("tool_action", None, rid, parent, {"tool": eff.kind, "args": eff.data, "seq": eff.seq,
                                                        "reversible": eff.reversible, "status": "applied"})

    def end_trace(self):
        """Mark the attempt complete and close the events file."""
        if self.tracer is not None:
            self._ev("episode_end", None, None, [], {"aborted": self.aborted})
            self.tracer.close()

    # ---------------- LLM plumbing ----------------
    def _account(self, role, stage, res):
        self.m["llm_calls"] += 1
        self.m["prompt_tokens"] += res.prompt_tokens
        self.m["completion_tokens"] += res.completion_tokens
        self.m["llm_time_s"] += res.latency_s
        self.calls.append({"role": role, "stage": stage, "prompt_tokens": res.prompt_tokens,
                           "completion_tokens": res.completion_tokens, "latency_s": res.latency_s,
                           "timings": res.extra.get("timings", {}), "finish": res.extra.get("finish"),
                           "reasoning_chars": res.extra.get("reasoning_chars", 0), "text": res.text})

    def _llm_json(self, role, system, user, meta):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(self.cfg.json_retries + 1):
            res = self.llm.chat(msgs, role=role, meta=dict(meta, retry=attempt))
            self._account(role, meta.get("stage"), res)
            run_id = meta.get("rng_key")
            self._msg_ev[run_id].append(self._ev(
                "message", meta.get("stage"), run_id, self._inputs.get(run_id, []),
                {"role": role, "text": res.text, "prompt_tokens": res.prompt_tokens,
                 "completion_tokens": res.completion_tokens, "latency_s": res.latency_s, "retry": attempt},
                prompt=msgs))
            obj = extract_json(res.text)
            if obj is not None:
                return obj
            self.m["json_failures"] += 1
            log.warning("%s: invalid JSON (attempt %d/%d, finish=%s): %s", role, attempt + 1,
                        self.cfg.json_retries + 1, res.extra.get("finish"), clip(res.text))
        log.error("%s: no valid JSON after %d attempts", role, self.cfg.json_retries + 1)
        return None

    # ---------------- stage pieces ----------------
    def _gather_inputs(self, stage, run: StageRun) -> dict:
        w = self.world
        if stage == "research_pricing":
            return {"task": w.task_text(), "tool_output": self.tools.list_prices()}
        if stage == "research_policy":
            return {"task": w.task_text(), "tool_output": self.tools.get_policy()}
        arts = {}
        for up in ARTIFACT_INPUTS[stage]:
            up_run = self.tracker.current[up]
            self.tracker.note_artifact_read(run, up_run)
            arts[up] = self.tracker.artifacts[up_run]
        if stage == "plan":
            vals = {}
            for key in PLAN_STATE_KEYS:
                value, writer = self.state.read_kv(run.run_id, key)
                self.tracker.note_state_read(run, key, writer)
                vals[key] = value
            return {"task": w.task_text(), "qty": w.qty, "sku": w.sku, "pricing": arts["research_pricing"],
                    "policy": {"approved": vals["approved_suppliers"],
                               "max_lead_days": vals["max_lead_days"], "notes": vals["policy_notes"]}}
        return {"plan": arts["plan"]}

    def _produce(self, stage, run, inputs) -> dict:
        system, user = agents.build_prompt(stage, inputs)
        meta = {"stage": stage, "inputs": inputs, "world_seed": self.world.seed, "rng_key": run.run_id}
        return agents.normalize(stage, self._llm_json(stage, system, user, meta))

    def _evidence(self, stage, inputs) -> dict:
        if stage == "research_pricing":
            return self.tools.list_prices()
        if stage == "research_policy":
            return self.tools.get_policy()
        if stage == "plan":
            return {"pricing": inputs["pricing"], "policy": inputs["policy"],
                    "qty": inputs["qty"], "sku": inputs["sku"]}
        return {"plan": inputs["plan"]}

    def _verify(self, stage, run, content, inputs) -> bool:
        evidence = self._evidence(stage, inputs)
        system, user = agents.build_verify_prompt(stage, content, evidence)
        meta = {"stage": stage, "artifact": content, "evidence": evidence,
                "world_seed": self.world.seed, "rng_key": run.run_id}
        obj = self._llm_json("verify", system, user, meta)
        self.m["verify_calls"] += 1
        if obj is None or "consistent" not in obj:
            self.m["verify_unparsed"] += 1
            log.warning("verify %s: unparseable verdict, defaulting to accept: %s", run.run_id, obj)
            return True
        ok = obj["consistent"] is True or str(obj["consistent"]).lower() == "true"
        log.info("verify %s -> %s (%s)", run.run_id, "accept" if ok else "REJECT", clip(obj.get("reason", ""), 120))
        return ok

    def _irreversible(self, kind, run_id, *args):
        """Apply an email/transfer now, or hold it in the outbox when the strategy is gated."""
        if self.gated:
            self.outbox.append((run_id, kind, args, self.m["llm_time_s"] + self.m["timeout_wasted_s"]))
            self._outbox_ev.append(self._ev("tool_action", None, run_id, [self._art_ev.get(run_id)], {
                "tool": kind.removeprefix("send_"), "args": dict(zip(EFFECT_ARGS[kind], args)),
                "reversible": False, "status": "held"}))
            return []
        return [getattr(self.state, kind)(run_id, *args)]

    def release_outbox(self, invalid_runs=()):
        """End of recovery: apply held effects whose run is not invalid. Emails are skipped when an identical
        (to, subject, body) email was already released. Uses only the invalidation set, never validate()."""
        if not self.outbox:
            return
        now = self.m["llm_time_s"] + self.m["timeout_wasted_s"]
        invalid = set(invalid_runs)
        for (rid, kind, args, _), held_ev in zip(self.outbox, self._outbox_ev):
            seq = None
            if rid in invalid:
                self.m["dropped_effects"] += 1
                status = "dropped"
            elif kind == "send_email" and args in self._sent_keys:
                self.m["idempotent_skips"] += 1
                status = "skipped"
            else:
                if kind == "send_email":
                    self._sent_keys.add(args)
                seq = getattr(self.state, kind)(rid, *args).seq
                self.m["released_effects"] += 1
                status = "released"
            self._ev("tool_action", None, rid, [self._art_ev.get(rid)], {
                "tool": kind.removeprefix("send_"), "args": dict(zip(EFFECT_ARGS[kind], args)), "seq": seq,
                "reversible": False, "status": status, "held_event": held_ev})
        self.m["held_effects"] = len(self.outbox)
        self.m["release_delay_s"] = now - self.outbox[0][3]
        log.info("outbox released: held=%d released=%d dropped=%d skips=%d", len(self.outbox),
                 self.m["released_effects"], self.m["dropped_effects"], self.m["idempotent_skips"])
        self.outbox, self._outbox_ev = [], []

    def _commit(self, stage, run, content):
        st, rid, w = self.state, run.run_id, self.world
        effs = []
        if stage == "research_policy":
            effs.append(st.set_kv(rid, "approved_suppliers", content.get("approved")))
            effs.append(st.set_kv(rid, "max_lead_days", content.get("max_lead_days")))
            effs.append(st.set_kv(rid, "policy_notes", content.get("notes", "")))
        elif stage == "execute":
            for act in content.get("actions", []):
                try:
                    tool, args = act["tool"], act.get("args") or {}
                    if tool == "place_order":
                        effs.append(st.place_order(rid, str(args["supplier"]), str(args["sku"]), int(args["qty"])))
                    elif tool == "update_ledger":
                        effs.append(st.update_ledger(rid, str(args["sku"]), int(args["delta"])))
                    elif tool == "transfer_funds":
                        effs += self._irreversible("transfer", rid, str(args["account"]), float(args["amount"]))
                    else:
                        self.m["invalid_tool_calls"] += 1
                        log.warning("%s: unknown tool %r", rid, tool)
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    self.m["invalid_tool_calls"] += 1
                    log.warning("%s: invalid action %s (%r)", rid, clip(act, 200), e)
        elif stage == "notify":
            sup = content.get("supplier")
            if isinstance(sup, str) and sup:
                effs += self._irreversible("send_email", rid, supplier_address(sup), f"Order confirmation {w.sku}",
                                           str(content.get("message", "")))
            else:
                self.m["invalid_tool_calls"] += 1
                log.warning("%s: notify has no supplier: %s", rid, clip(content, 200))
            if content.get("cc"):
                effs += self._irreversible("send_email", rid, str(content["cc"]), f"Order confirmation {w.sku}",
                                           str(content.get("message", "")))
        run.effect_seqs.extend(e.seq for e in effs)
        for e in effs:
            log.debug("%s: effect #%d %s %s reversible=%s", rid, e.seq, e.kind, clip(e.data, 200), e.reversible)
            self._trace_effect(e)

    # ---------------- public stage execution ----------------
    def _count_detection(self, run_id, rejected):
        injected = run_id == self.fault_run_id
        self.m[("det_tp" if injected else "det_fp") if rejected else ("det_fn" if injected else "det_tn")] += 1

    def execute_stage(self, stage, verify=False, force_commit=False, rules=False):
        """Run one stage. Returns the StageRun, or None if the verifier (or, with rules=True, the rule checker)
        rejected the output. With rules=True the rule checker runs before the LLM verifier; on the last attempt
        (force_commit) a rule-breaking output is committed with the violating parts removed (D28).
        Raises StageTimeout if an injected timeout fires."""
        self.attempts[stage] += 1
        run_id = f"{stage}#{self.attempts[stage]}"
        run = StageRun(run_id, stage, self.attempts[stage], is_recovery=self.in_recovery)
        self.tracker.runs[run_id] = run
        log.info("start %s world_seed=%s recovery=%s verify=%s", run_id, self.world.seed, self.in_recovery, verify)
        fires = self.fault is not None and not self.fault_fired and self.fault.stage == stage
        if fires and self.fault.kind == "timeout":
            self.fault_fired, self.fault_run_id = True, run_id
            run.status = "timeout"
            self.m["timeout_wasted_s"] += self.cfg.timeout_s
            log.info("%s: injected TIMEOUT (charged %.0fs)", run_id, self.cfg.timeout_s)
            self._ev("fault", stage, run_id, [], {"fault": "timeout", "timeout_s": self.cfg.timeout_s})
            raise StageTimeout(run_id)
        inputs = self._gather_inputs(stage, run)
        self._trace_reads(run)
        content = self._produce(stage, run, inputs)
        fault_ev = None
        if fires:
            content = faults.inject(self.fault, content, self.world)
            self.fault_fired, self.fault_run_id = True, run_id
            self.fault_noop = self.fault.kind not in faults.PROBE_STAGES and validate(stage, content, self.world)  # probes keep the answer on purpose
            fault_ev = self._ev("fault", stage, run_id, self._msg_ev[run_id][-1:],
                                {"fault": self.fault.kind, "noop": self.fault_noop})
            log.info("%s: injected fault %s@%s noop=%s -> %s", run_id, self.fault.kind, stage,
                        self.fault_noop, clip(content, 300))
        violations = []
        if rules:
            plan = inputs.get("plan")
            violations = rule_checker.check(stage, content, self.world.sku, self.world.qty,
                                            plan.get("supplier") if isinstance(plan, dict) else None)
        if violations:
            self.m["rule_rejections"] += 1
            self._count_detection(run_id, True)
            log.info("rules %s -> REJECT %s", run_id, clip(violations, 300))
            self._ev("verify", stage, run_id, self._inputs.get(run_id, []),
                     {"verdict": "reject", "checker": "rules", "violations": violations, "forced": force_commit})
            if not force_commit:
                run.status = "rejected"
                log.info("%s: rejected by rules, not committed", run_id)
                return None
            content, n = rule_checker.remove_violations(content, violations)
            self.m["blocked_actions"] += n
            log.warning("%s: rules rejected again, committed with %d violating part(s) removed", run_id, n)
        elif verify:
            if rules:
                self._ev("verify", stage, run_id, self._inputs.get(run_id, []),
                         {"verdict": "accept", "checker": "rules"})
            unparsed = self.m["verify_unparsed"]
            ok = self._verify(stage, run, content, inputs)
            self._ev("verify", stage, run_id, self._inputs.get(run_id, []),
                     {"verdict": "accept" if ok else "reject", "checker": "llm", "forced": not ok and force_commit,
                      "unparsed": self.m["verify_unparsed"] > unparsed})
            self._count_detection(run_id, not ok)
            if not ok and rules:
                self.m["llm_rejections"] += 1
            if not ok and not force_commit:
                run.status = "rejected"
                log.info("%s: rejected by verifier, not committed", run_id)
                return None
            if not ok:
                log.warning("%s: rejected by verifier but force-committed (attempts exhausted)", run_id)
        self._art_ev[run_id] = self._ev("artifact", stage, run_id, self._inputs.get(run_id, []) + ([fault_ev] if fault_ev else []), {
            "content": content, "recovery": self.in_recovery, "message_ids": self._msg_ev[run_id],
            "fault_event": fault_ev})
        self._commit(stage, run, content)
        self.tracker.artifacts[run_id] = content
        self.tracker.current[stage] = run_id
        run.status = "ok"
        log.info("done %s status=ok effects=%d", run_id, len(run.effect_seqs))
        return run

    def rollback(self, run_ids):
        """Undo the reversible effects of the given runs (newest first) and retire their artifacts."""
        seqs = sorted((s for rid in run_ids for s in self.tracker.runs[rid].effect_seqs), reverse=True)
        log.info("rollback runs=%s effects=%d", list(run_ids), len(seqs))
        stuck = 0
        for s in seqs:
            eff = self.state.effects[s]
            if not eff.undone and not self.state.undo(eff):
                self.m["irreversible_not_undone"] += 1
                stuck += 1
                log.info("rollback: effect #%d %s by %s is irreversible and stays", s, eff.kind, eff.run_id)
        self._ev("rollback", None, None, [self._art_ev[r] for r in run_ids if r in self._art_ev],
                 {"runs": list(run_ids), "effects": len(seqs), "irreversible_not_undone": stuck})
        for rid in run_ids:
            run = self.tracker.runs[rid]
            if run.status == "ok":
                run.status = "superseded"
            if self.tracker.current.get(run.stage) == rid:
                del self.tracker.current[run.stage]

    def mark_first_pass_done(self):
        """Count downstream artifacts that are invalid in the first pass (= fault propagation)."""
        n = 0
        for stage, rid in self.tracker.current.items():
            if rid != self.fault_run_id and not validate(stage, self.tracker.artifacts[rid], self.world):
                n += 1
        self.m["propagated_invalid"] = n
        log.info("first pass done: propagated_invalid=%d fault_fired=%s fault_run=%s", n, self.fault_fired,
                 self.fault_run_id)

    # ---------------- results ----------------
    def summary(self) -> dict:
        runs = list(self.tracker.runs.values())
        live = [self.tracker.runs[r] for r in self.tracker.current.values()]
        retained = [r for r in live if not r.is_recovery]
        valid = lambda r: validate(r.stage, self.tracker.artifacts[r.run_id], self.world)
        m = self.m
        out = evaluate_final(self.world, self.state)
        out.update({
            "fault_fired": self.fault_fired, "fault_noop": self.fault_noop, "aborted": self.aborted,
            "stage_runs": len(runs),
            "recomputed_stages": sum(1 for r in runs if r.is_recovery),
            "wasted_stage_runs": sum(1 for r in runs if r.status in ("superseded", "rejected", "timeout")),
            "retained_valid": sum(1 for r in retained if valid(r)),
            "retained_invalid": sum(1 for r in retained if not valid(r)),
            "propagated_invalid": int(m["propagated_invalid"]),
            "irreversible_not_undone": int(m["irreversible_not_undone"]),
            "llm_calls": int(m["llm_calls"]), "verify_calls": int(m["verify_calls"]),
            "verify_unparsed": int(m["verify_unparsed"]), "json_failures": int(m["json_failures"]),
            "invalid_tool_calls": int(m["invalid_tool_calls"]),
            "prompt_tokens": int(m["prompt_tokens"]), "completion_tokens": int(m["completion_tokens"]),
            "llm_time_s": m["llm_time_s"], "timeout_wasted_s": m["timeout_wasted_s"],
            "latency_s": m["llm_time_s"] + m["timeout_wasted_s"],
            "tracking_overhead_s": self.tracker.time_s,
            "trace_overhead_s": self.tracer.time_s if self.tracer else 0.0,
            "held_effects": int(m["held_effects"]), "released_effects": int(m["released_effects"]),
            "dropped_effects": int(m["dropped_effects"]), "idempotent_skips": int(m["idempotent_skips"]),
            "release_delay_s": m["release_delay_s"],
            "det_tp": int(m["det_tp"]), "det_fp": int(m["det_fp"]),
            "det_fn": int(m["det_fn"]), "det_tn": int(m["det_tn"]),
            "rule_rejections": int(m["rule_rejections"]), "llm_rejections": int(m["llm_rejections"]),
            "blocked_actions": int(m["blocked_actions"]),
        })
        return out

    def trace(self) -> dict:
        return {
            "runs": [{"run_id": r.run_id, "stage": r.stage, "status": r.status, "recovery": r.is_recovery,
                      "reads": r.read_artifacts, "state_reads": r.read_state, "effects": r.effect_seqs,
                      "artifact": self.tracker.artifacts.get(r.run_id)} for r in self.tracker.runs.values()],
            "effects": [{"seq": e.seq, "run": e.run_id, "kind": e.kind, "data": e.data,
                         "reversible": e.reversible, "undone": e.undone} for e in self.state.effects],
            "calls": self.calls,
        }
