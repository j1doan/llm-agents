"""Dependency oracle (D26, D29) and probe faults. Mock backend only: this checks the harness, not model behaviour."""
import json
import os
import tempfile
import unittest

from maf import oracle
from maf.experiment import build_grid, main as experiment_main, run_episode
from maf.faults import FAULT_STAGES, PROBE_STAGES, FaultSpec, inject
from maf.llm import MockLLM
from maf.strategies import STRATEGIES
from maf.workflow import Config
from maf.world import make_world, solve_plan, tool_list_prices

LLM = MockLLM(obey_prob=1.0)


def run(path, seed, kind=None, stage=None, strategy="none"):
    fault = FaultSpec(kind, stage) if kind else None
    rec, _ = run_episode(LLM, "mock", seed, fault, strategy, Config(), events_path=path + ".events.jsonl")
    return rec


def tmp():
    return os.path.join(tempfile.mkdtemp(), "r.jsonl")


class TrueAndPredicted(unittest.TestCase):
    def setUp(self):
        self.path = tmp()
        run(self.path, 0, None)
        for stage in ("research_pricing", "research_policy", "plan"):
            run(self.path, 0, "wrong_result", stage)
        self.eps, _ = oracle.load_episodes(self.path + ".events.jsonl")

    def obs(self, stage):
        return oracle.observe(self.eps["mock|0|clean|none"], self.eps[f"mock|0|wrong_result@{stage}|none"])

    def test_plan_fault(self):
        o = self.obs("plan")
        want = {"execute", "notify"}      # the faulted stage itself is excluded (D36)
        self.assertEqual(o["pred"][True], want)
        self.assertEqual(o["true"]["strict"], want)
        self.assertEqual(o["true"]["decision"], want)

    def test_pricing_fault_reaches_downstream_by_messages(self):
        o = self.obs("research_pricing")
        self.assertEqual(o["pred"][False], {"plan", "execute", "notify"})
        self.assertEqual(o["pred"][True], o["pred"][False])

    def test_message_only_misses_shared_state_dependency(self):
        o = self.obs("research_policy")
        self.assertIn("plan", o["pred"][True])
        self.assertEqual(o["pred"][False], set())
        self.assertIn("plan", o["true"]["strict"])      # the miss is real: plan did change

    def test_cross_check_against_recovery_rollback(self):
        for name in ("selective", "selective_msg"):
            run(self.path, 0, "wrong_result", "research_policy", strategy=name)
        run(self.path, 0, "timeout", "plan")
        obs, cross, skipped = oracle.collect(self.path)
        self.assertEqual(cross[True][:2], [1, 0])
        self.assertEqual(cross[False][:2], [1, 0])
        self.assertEqual(skipped["timeout"], 1)
        self.assertEqual(len(obs), 3)       # the timeout is skipped; selective episodes are not observations

    def test_incomplete_attempt_and_error_record_are_skipped_and_counted(self):
        with open(self.path + ".events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "e1", "episode": "mock|1|clean|none", "attempt": 1, "stage": None,
                                "run_id": None, "kind": "fault", "parents": [], "t": 0, "payload": {}}) + "\n")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "mock|1|clean|none", "error": "boom"}) + "\n")
        _, _, skipped = oracle.collect(self.path)
        self.assertEqual((skipped["incomplete_attempts"], skipped["error_records"]), (1, 1))

    def test_scores(self):
        obs = [self.obs(s) for s in ("research_pricing", "research_policy", "plan")]
        rows = {(r["tracker"], r["notion"]): r for r in oracle.score_table(obs) if r["level"] == "pooled"}
        full = rows[("selective", "strict")]
        self.assertEqual((full["precision"], full["recall"]), (1.0, 1.0))
        msg = rows[("selective_msg", "strict")]
        self.assertEqual(msg["precision"], 1.0)
        self.assertLess(msg["recall"], 1.0)

    def test_empty_sets_are_counted_and_score_na(self):
        o = self.obs("research_policy")         # selective_msg predicts nothing downstream
        row = oracle.score([o], False, "strict")
        self.assertEqual((row["n"], row["pred"]), (1, 0))
        self.assertNotEqual(row["precision"], row["precision"])     # nan
        self.assertEqual(row["recall"], 0.0)
        probe = {"kind": "probe_notes", "stage": "plan", "pred": {True: {"execute"}, False: set()},
                 "true": {"strict": set(), "decision": set()}}
        row = oracle.score([probe], True, "decision")
        self.assertEqual((row["n"], row["precision"]), (1, 0.0))
        self.assertNotEqual(row["recall"], row["recall"])


class StrictVsDecision(unittest.TestCase):
    def test_rationale_change_is_strict_only(self):
        base = {"plan": {"supplier": "A", "sku": "S", "qty": 5, "rationale": "x"},
                "notify": {"supplier": "A", "message": "hi"}}
        other = {"plan": {"supplier": "A", "sku": "S", "qty": 5, "rationale": "y"},
                 "notify": {"supplier": "A", "message": "hi", "cc": "e@evil.test"}}
        self.assertEqual(oracle.true_set(base, other, "research_pricing", "strict"),
                         {"plan", "notify"})
        self.assertEqual(oracle.true_set(base, other, "research_pricing", "decision"),
                         {"notify"})

    def test_upstream_difference_is_not_in_true_set(self):
        base = {"research_pricing": {"p": 1}, "plan": {"supplier": "A", "qty": 5}, "notify": {"supplier": "A"}}
        other = {"research_pricing": {"p": 2}, "plan": {"supplier": "B", "qty": 5}, "notify": {"supplier": "A"}}
        for notion in ("strict", "decision"):
            self.assertEqual(oracle.true_set(base, other, "plan", notion), set())
        self.assertEqual(oracle.true_set(base, other, "research_pricing", "decision"), {"plan"})

    def test_execute_decision_uses_tool_and_args(self):
        a = {"actions": [{"tool": "place_order", "args": {"qty": 1}}]}
        b = {"actions": [{"tool": "place_order", "args": {"qty": 10}}]}
        self.assertNotEqual(oracle.signature("execute", a, "decision"), oracle.signature("execute", b, "decision"))


class Probes(unittest.TestCase):
    def test_probes_keep_the_answer_and_are_not_noop(self):
        for kind, stages in PROBE_STAGES.items():
            for seed in range(5):
                rec = run(tmp(), seed, kind, stages[0])
                self.assertTrue(rec["fault_fired"], (kind, seed))
                self.assertFalse(rec["fault_noop"], (kind, seed))
                self.assertTrue(rec["clean_success"], (kind, seed))

    def test_price_probe_raises_only_an_ineligible_supplier(self):
        for seed in range(10):
            w = make_world(seed)
            before = tool_list_prices(w)
            after = inject(FaultSpec("probe_price", "research_pricing"), before, w)
            changed = [o["supplier"] for o, p in zip(after["offers"], before["offers"]) if o != p]
            self.assertEqual(len(changed), 1)
            self.assertNotIn(changed[0], w.feasible())
            pol = {"approved": w.approved, "max_lead_days": w.max_lead_days}
            self.assertEqual(solve_plan(after, pol, w.qty, w.sku)["supplier"], w.expected_supplier)

    def test_text_probes_change_only_their_field(self):
        def rest(d, k):
            return {a: b for a, b in d.items() if a != k}
        c = {"approved": ["A"], "max_lead_days": 5, "notes": ""}
        out = inject(FaultSpec("probe_notes", "research_policy"), c, make_world(0))
        self.assertEqual(rest(out, "notes"), rest(c, "notes"))
        self.assertNotEqual(out["notes"], c["notes"])
        p = {"supplier": "A", "sku": "S", "qty": 1, "rationale": "r"}
        out = inject(FaultSpec("probe_rationale", "plan"), p, make_world(0))
        self.assertEqual(rest(out, "rationale"), rest(p, "rationale"))
        self.assertNotEqual(out["rationale"], "r")

    def test_probes_never_enter_the_default_grid(self):
        self.assertEqual(set(FAULT_STAGES), {"wrong_result", "malicious", "stale", "timeout"})
        grid = build_grid([0], list(FAULT_STAGES), list(STRATEGIES))
        self.assertTrue(all(not (f and f.kind.startswith("probe_")) for _, f, _ in grid))
        self.assertEqual(len(grid), (1 + sum(len(v) for v in FAULT_STAGES.values())) * len(STRATEGIES))
        with self.assertRaises(SystemExit):
            experiment_main(["--kinds", "probe_price", "--estimate"])

    def test_probe_command_then_collect(self):
        path = tmp()
        oracle.main(["probe", "--backend", "mock", "--seeds", "0", "--out", path, "--log-dir", ""])
        obs, _, skipped = oracle.collect(path)
        self.assertEqual({o["kind"] for o in obs}, set(PROBE_STAGES))
        self.assertEqual(sum(skipped.values()), 0)
        for o in obs:
            self.assertEqual(o["pred"][True] & {"execute", "notify"}, {"execute", "notify"})
            self.assertEqual(o["true"]["decision"], set())       # probes change nothing downstream


if __name__ == "__main__":
    unittest.main()
