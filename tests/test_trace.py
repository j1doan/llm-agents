"""Per-event trace (D22), pin wiring (D13) and analyze coverage of all strategies. Mock backend only."""
import contextlib
import io
import json
import os
import tempfile
import unittest

from maf import analyze
from maf.experiment import main, run_episode
from maf.faults import FaultSpec
from maf.llm import MockLLM
from maf.strategies import STRATEGIES
from maf.trace import EventTrace, load_attempts
from maf.workflow import Config, Episode
from maf.world import make_world

LLM = MockLLM(obey_prob=1.0)


def run_traced(strategy, fault, seed=0, attempt=1):
    """Returns (episode, events) for one traced mock episode."""
    path = os.path.join(tempfile.mkdtemp(), "e.jsonl")
    key = f"mock|{seed}|x|{strategy}"
    ep = Episode(make_world(seed), LLM, fault, Config(), EventTrace(path, key, attempt))
    STRATEGIES[strategy](ep)
    ep.end_trace()
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    return ep, events


class EventTests(unittest.TestCase):
    def test_artifact_parents_match_tracker_reads(self):
        ep, ev = run_traced("selective", FaultSpec("wrong_result", "research_policy"))
        art = {e["run_id"]: e["id"] for e in ev if e["kind"] == "artifact"}
        writes = {(e["run_id"], e["payload"]["key"]): e["id"] for e in ev if e["kind"] == "state_write"}
        checked = 0
        for run in ep.tracker.runs.values():
            if run.run_id not in art:
                continue
            want = {art[a] for a in run.read_artifacts} | {writes[(w, k)] for k, w in run.read_state if w}
            got = next(e["parents"] for e in ev if e["id"] == art[run.run_id])
            got = [p for p in got if next(e for e in ev if e["id"] == p)["kind"] != "fault"]   # fault parent: T13
            self.assertEqual(set(got), want, run.run_id)
            checked += bool(want)
        self.assertGreaterEqual(checked, 3)    # plan, execute, notify read something
        plan = next(e for e in ev if e["kind"] == "artifact" and e["stage"] == "plan")
        self.assertTrue(any(e["kind"] == "state_write" and e["id"] in plan["parents"] for e in ev))

    def test_faulted_artifact_lists_the_fault_event_as_parent(self):
        _, ev = run_traced("none", FaultSpec("wrong_result", "research_policy"))
        fault = next(e for e in ev if e["kind"] == "fault")
        art = next(e for e in ev if e["kind"] == "artifact" and e["run_id"] == fault["run_id"])
        self.assertIn(fault["id"], art["parents"])
        self.assertEqual(art["payload"]["fault_event"], fault["id"])
        for e in ev:        # no other event points at the fault
            if e["kind"] == "artifact" and e is not art:
                self.assertNotIn(fault["id"], e["parents"])

    def test_tool_actions_have_artifact_parent_and_all_kinds_present(self):
        kinds = set()
        for strategy in STRATEGIES:
            for fault in (None, FaultSpec("malicious", "plan"), FaultSpec("timeout", "execute")):
                _, ev = run_traced(strategy, fault)
                art_ids = {e["id"] for e in ev if e["kind"] == "artifact"}
                kinds |= {e["kind"] for e in ev}
                self.assertEqual(ev[-1]["kind"], "episode_end")
                for e in ev:
                    self.assertTrue(set(e["parents"]) <= {x["id"] for x in ev})
                    if e["kind"] == "tool_action":
                        self.assertTrue(set(e["parents"]) & art_ids, (strategy, e))
        self.assertEqual(kinds, {"message", "artifact", "fault", "state_read", "state_write", "tool_action",
                                 "verify", "rollback", "episode_end"})

    def test_gated_episode_holds_then_releases_or_drops(self):
        _, ev = run_traced("selective_gated", FaultSpec("wrong_result", "plan"))
        status = [e["payload"]["status"] for e in ev if e["kind"] == "tool_action"
                  and not e["payload"]["reversible"]]
        self.assertIn("held", status)
        self.assertTrue({"released", "dropped"} & set(status))
        self.assertLess(status.index("held"), min(i for i, s in enumerate(status) if s in ("released", "dropped")))
        held = {e["id"] for e in ev if e["payload"].get("status") == "held"}
        for e in ev:
            if e["payload"].get("status") in ("released", "dropped", "skipped"):
                self.assertIn(e["payload"]["held_event"], held)

    def test_attempts_and_hash_not_prompt(self):
        _, ev = run_traced("none", None, attempt=3)
        self.assertEqual({e["attempt"] for e in ev}, {3})
        msg = next(e for e in ev if e["kind"] == "message")
        self.assertEqual(len(msg["payload"]["prompt_sha256"]), 64)
        self.assertNotIn("prompt", msg["payload"])

    def test_load_attempts(self):
        path = os.path.join(tempfile.mkdtemp(), "e.jsonl")
        EventTrace(path, "k", 2).emit("fault", None, None, [], 0, {})
        self.assertEqual(load_attempts(path), {"k": 2})
        self.assertEqual(load_attempts(path + ".none"), {})


class GridTests(unittest.TestCase):
    def grid(self, *extra):
        d = tempfile.mkdtemp()
        out = os.path.join(d, "r.jsonl").replace("\\", "/")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--backend", "mock", "--seeds", "0", "--kinds", "wrong_result", "--strategies",
                  "selective_gated,none", "--log-dir", "", "--out", out, *extra])
        with open(out, encoding="utf-8") as f:
            return d, out, [json.loads(line) for line in f]

    def test_trace_on_by_default_and_pins_written(self):
        d, out, recs = self.grid()
        self.assertTrue(os.path.exists(out + ".events.jsonl"))
        self.assertTrue(all(r["trace_overhead_s"] > 0 for r in recs))
        pins_text = open(out + ".pins.json", encoding="utf-8").read()
        pins = json.loads(pins_text)
        self.assertEqual(pins["model_settings"]["max_output_tokens"], 512)
        self.assertNotIn(d.replace("\\", "/"), pins_text.replace("\\\\", "/"))
        self.assertNotIn(os.path.expanduser("~").replace("\\", "/"), pins_text.replace("\\\\", "/"))
        # rerun resumes: nothing to do, same pins, no new events
        size = os.path.getsize(out + ".events.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--backend", "mock", "--seeds", "0", "--kinds", "wrong_result", "--strategies",
                  "selective_gated,none", "--log-dir", "", "--out", out])
        self.assertEqual(os.path.getsize(out + ".events.jsonl"), size)

    def test_no_trace(self):
        _, out, recs = self.grid("--no-trace")
        self.assertFalse(os.path.exists(out + ".events.jsonl"))
        self.assertTrue(all(r["trace_overhead_s"] == 0 for r in recs))

    def test_trace_does_not_change_outcomes(self):
        _, _, on = self.grid()
        _, _, off = self.grid("--no-trace")
        skip = {"wall_time_s", "trace_overhead_s", "tracking_overhead_s"}
        strip = lambda r: {k: v for k, v in r.items() if k not in skip}
        self.assertEqual([strip(r) for r in on], [strip(r) for r in off])

    def test_save_traces_old_format_still_works(self):
        _, out, _ = self.grid("--save-traces")
        with open(out + ".traces.jsonl", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        self.assertEqual(set(rec), {"key", "runs", "effects", "calls"})

    def test_run_episode_without_events_path(self):
        rec, _ = run_episode(LLM, "mock", 0, None, "none", Config())
        self.assertEqual(rec["trace_overhead_s"], 0.0)


class AnalyzeTests(unittest.TestCase):
    def test_all_strategies_listed_and_hybrid_detection(self):
        rows = [run_episode(LLM, "mock", 0, FaultSpec("malicious", "plan"), s, Config())[0] for s in STRATEGIES]
        self.assertEqual(set(analyze.STRATEGY_ORDER), set(STRATEGIES))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            md, _, _ = analyze.table(rows, "kind", analyze.COLUMNS)
            print(md)
            analyze.detection(rows)
        for s in STRATEGIES:
            self.assertIn(f"| {s} |", buf.getvalue())
        self.assertIn("verify_hybrid detection", buf.getvalue())
        self.assertIn("verify_all detection", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
