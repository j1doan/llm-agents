"""Run with:  python -m pytest -q  (from the project root; python -m unittest discover -s tests -t . also works)"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from maf.agents import normalize
from maf.experiment import build_grid, episode_key, parse_seeds, run_episode
from maf.faults import FAULT_STAGES, FaultSpec
from maf.llm import LlamaServerClient, MockLLM
from maf.strategies import STRATEGIES
from maf.util import extract_json
from maf.workflow import Config, Episode
from maf.world import make_world

LLM = MockLLM(obey_prob=1.0)
CFG = Config()


def run(seed, strategy, kind=None, stage=None, llm=LLM):
    fault = FaultSpec(kind, stage) if kind else None
    rec, _ = run_episode(llm, "mock", seed, fault, strategy, CFG)
    return rec


class WorldTests(unittest.TestCase):
    def test_world_is_deterministic_and_nontrivial(self):
        for seed in range(30):
            w = make_world(seed)
            self.assertEqual(w.expected_supplier, make_world(seed).expected_supplier)
            cheapest = min(w.offers, key=lambda n: w.offers[n]["price"])
            self.assertNotIn(cheapest, w.feasible())
            self.assertGreaterEqual(len(w.feasible()), 2)

    def test_extract_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('noise ```json\n{"a": {"b": 2}}\n``` tail'), {"a": {"b": 2}})
        self.assertEqual(extract_json('sure! {"a": "}"} done'), {"a": "}"})
        self.assertIsNone(extract_json("no json here"))


class Normalize(unittest.TestCase):
    def test_plan_drops_scratch_fields(self):
        raw = {"max_lead_days": 7, "checks": [{"supplier": "A", "price": 1.0, "lead_days": 2, "within_limit": True}],
               "supplier": "A", "sku": "S", "qty": 5, "rationale": "r", "extra": 1}
        self.assertEqual(normalize("plan", raw), {"supplier": "A", "sku": "S", "qty": 5, "rationale": "r"})


class CleanRuns(unittest.TestCase):
    def test_clean_runs_succeed_for_every_strategy(self):
        for seed in range(5):
            for strategy in STRATEGIES:
                r = run(seed, strategy)
                self.assertTrue(r["clean_success"], (seed, strategy, r))
                self.assertEqual(r["recomputed_stages"], 0)
                self.assertEqual(r["unauthorized_ever"], 0)

    def test_verification_costs_extra_calls(self):
        self.assertEqual(run(0, "none")["llm_calls"], 5)
        self.assertEqual(run(0, "verify_all")["llm_calls"], 10)


class FaultPropagation(unittest.TestCase):
    def test_every_supported_fault_is_effective_without_recovery(self):
        for kind, stages in FAULT_STAGES.items():
            for stage in stages:
                for seed in range(5):
                    r = run(seed, "none", kind, stage)
                    self.assertTrue(r["fault_fired"])
                    self.assertFalse(r["fault_noop"], (kind, stage, seed))
                    if kind in ("wrong_result", "stale"):
                        self.assertFalse(r["task_success"], (kind, stage, seed))

    def test_upstream_fault_propagates_downstream(self):
        r = run(1, "none", "wrong_result", "research_pricing")
        self.assertEqual(r["propagated_invalid"], 3)        # plan, execute, notify
        r = run(1, "none", "wrong_result", "research_policy")
        self.assertEqual(r["propagated_invalid"], 3)        # via shared state, not messages

    def test_injection_reaches_executor_through_the_planner(self):
        r = run(2, "none", "malicious", "research_pricing")
        self.assertEqual(r["unauthorized_final"], 1)
        r = run(2, "none", "malicious", "research_policy")  # also flows through shared state
        self.assertEqual(r["unauthorized_final"], 1)


class Recovery(unittest.TestCase):
    def test_selective_reuses_valid_work(self):
        r = run(3, "selective", "wrong_result", "research_pricing")
        self.assertTrue(r["task_success"])
        self.assertEqual(r["recomputed_stages"], 4)         # everything but research_policy
        self.assertEqual(r["retained_valid"], 1)
        self.assertEqual(r["retained_invalid"], 0)
        # the wrong email already went out before the fault was identified
        self.assertFalse(r["clean_success"])
        self.assertEqual(r["wrong_irreversible"], 1)

    def test_selective_beats_full_restart_on_recompute(self):
        sel = run(3, "selective", "wrong_result", "execute")
        full = run(3, "full_restart", "wrong_result", "execute")
        self.assertEqual(sel["recomputed_stages"], 1)
        self.assertEqual(full["recomputed_stages"], 5)
        self.assertTrue(sel["clean_success"])
        self.assertTrue(full["task_success"])
        self.assertEqual(full["duplicate_irreversible"], 1)  # restart resends the email
        self.assertFalse(full["clean_success"])

    def test_message_only_tracking_misses_shared_state_dependency(self):
        good = run(4, "selective", "wrong_result", "research_policy")
        bad = run(4, "selective_msg", "wrong_result", "research_policy")
        self.assertTrue(good["task_success"])
        self.assertFalse(bad["task_success"])
        self.assertGreaterEqual(bad["retained_invalid"], 1)  # the stale plan was reused

    def test_verifier_stops_faults_before_they_commit(self):
        for stage in ("research_pricing", "research_policy", "plan", "execute", "notify"):
            r = run(5, "verify_all", "wrong_result", stage)
            self.assertTrue(r["clean_success"], stage)
            self.assertEqual(r["det_tp"], 1)
            self.assertEqual(r["det_fn"], 0)
        r = run(5, "verify_all", "malicious", "execute")
        self.assertEqual(r["unauthorized_ever"], 0)

    def test_irreversible_injected_actions_cannot_be_undone(self):
        r = run(6, "selective", "malicious", "research_pricing")
        self.assertTrue(r["task_success"])
        self.assertEqual(r["unauthorized_final"], 1)         # the transfer survives recovery
        self.assertGreaterEqual(r["irreversible_not_undone"], 1)

    def test_gate_drops_wrong_email_after_notify_fault(self):
        ungated = run(0, "full_restart", "wrong_result", "notify")
        self.assertEqual(ungated["wrong_irreversible"], 1)           # the wrong email already went out
        self.assertEqual(ungated["held_effects"], 0)
        r = run(0, "full_restart_gated", "wrong_result", "notify")
        self.assertTrue(r["clean_success"])
        self.assertEqual((r["wrong_irreversible"], r["duplicate_irreversible"]), (0, 0))
        self.assertEqual((r["held_effects"], r["released_effects"], r["dropped_effects"]), (2, 1, 1))
        self.assertGreater(r["release_delay_s"], 0)

    def test_selective_gate_releases_no_wrong_email_after_upstream_fault(self):
        for stage in ("research_pricing", "research_policy", "plan"):
            r = run(0, "selective_gated", "wrong_result", stage)
            self.assertTrue(r["clean_success"], stage)
            self.assertEqual((r["released_effects"], r["dropped_effects"]), (1, 1), stage)

    def test_gated_clean_run_matches_twin(self):
        for gated, twin in (("full_restart_gated", "full_restart"), ("selective_gated", "selective")):
            a, b = run(0, gated), run(0, twin)
            self.assertEqual((a["task_success"], a["clean_success"]), (b["task_success"], b["clean_success"]))
            self.assertEqual((a["held_effects"], a["released_effects"], a["dropped_effects"]), (1, 1, 0))

    def test_gate_skips_duplicate_email(self):
        ep = Episode(make_world(0), LLM)
        ep.gated = True
        for rid in ("notify#1", "notify#2"):
            ep._irreversible("send_email", rid, "a@x.com", "s", "body")
        ep.release_outbox()
        self.assertEqual((len(ep.state.emails), ep.m["idempotent_skips"]), (1, 1))

    def test_timeouts(self):
        self.assertFalse(run(7, "none", "timeout", "execute")["task_success"])
        sel = run(7, "selective", "timeout", "execute")
        self.assertTrue(sel["clean_success"])
        self.assertEqual(sel["recomputed_stages"], 1)
        full = run(7, "full_restart", "timeout", "execute")
        self.assertTrue(full["clean_success"])
        self.assertEqual(full["recomputed_stages"], 5)
        ver = run(7, "verify_all", "timeout", "execute")
        self.assertTrue(ver["clean_success"])

    def test_all_strategies_survive_all_faults(self):
        for strategy in STRATEGIES:
            for kind, stages in FAULT_STAGES.items():
                for stage in stages:
                    r = run(8, strategy, kind, stage)
                    self.assertTrue(r["fault_fired"])
                    self.assertGreater(r["stage_runs"], 0)


class Experiment(unittest.TestCase):
    def test_grid_and_keys(self):
        grid = build_grid(parse_seeds("0-1"), list(FAULT_STAGES), list(STRATEGIES))
        per_seed = (1 + sum(len(v) for v in FAULT_STAGES.values())) * len(STRATEGIES)
        self.assertEqual(len(grid), 2 * per_seed)
        keys = [episode_key("m", s, f, st) for s, f, st in grid]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(parse_seeds("0-2,7"), [0, 1, 2, 7])


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append(body)
        out = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}],
                          "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                          "timings": {"predicted_per_second": 9.5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class LlamaClient(unittest.TestCase):
    def test_request_and_response_plumbing(self):
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            client = LlamaServerClient(f"http://127.0.0.1:{srv.server_address[1]}", fold_system=True)
            res = client.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(res.text, '{"ok": true}')
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (11, 3))
        self.assertEqual(res.extra["timings"]["predicted_per_second"], 9.5)
        sent = _Handler.seen[-1]
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["messages"], [{"role": "user", "content": "S\n\nU"}])


if __name__ == "__main__":
    unittest.main()
