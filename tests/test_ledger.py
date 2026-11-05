"""SQLite-backed orders and ledger (D37)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from maf import experiment
from maf.experiment import run_episode
from maf.llm import MockLLM
from maf.state import SimState
from maf.workflow import Config

LLM = MockLLM(obey_prob=1.0)


def rows(state, table):
    return state._db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


class SqliteStateTests(unittest.TestCase):
    def setUp(self):
        self.s = SimState(["A"], "sqlite")
        self.addCleanup(self.s.close)

    def test_writes_are_in_the_db(self):
        self.s.place_order("r1", "sup", "A", 5)
        self.s.update_ledger("r1", "A", 5)
        self.assertEqual(rows(self.s, "orders"), [(1, "sup", "A", 5)])
        self.assertEqual(rows(self.s, "ledger"), [("A", 5)])
        self.assertEqual(self.s.orders, [{"id": 1, "supplier": "sup", "sku": "A", "qty": 5}])
        self.assertEqual(self.s.ledger, {"A": 5})
        self.assertGreater(self.s.io_s, 0)

    def test_undo_reverses_in_the_db(self):
        o = self.s.place_order("r1", "sup", "A", 5)
        l = self.s.update_ledger("r1", "A", 5)
        self.assertTrue(self.s.undo(o) and self.s.undo(l))
        self.assertEqual(rows(self.s, "orders"), [])
        self.assertEqual(rows(self.s, "ledger"), [("A", 0)])

    def test_same_as_memory(self):
        m = SimState(["A"])
        for st in (m, self.s):
            e = st.place_order("r", "s", "B", 2)
            st.update_ledger("r", "B", 3)
            st.place_order("r", "s", "A", 1)
            st.undo(e)
        self.assertEqual((m.orders, m.ledger), (self.s.orders, self.s.ledger))
        self.assertEqual(m.io_s, 0.0)

    def test_close_deletes_file(self):
        path = self.s._tmp
        self.assertTrue(os.path.isdir(path))
        self.s.close()
        self.s.close()
        self.assertFalse(os.path.exists(path))


class EpisodeTests(unittest.TestCase):
    def test_file_deleted_after_error_and_record_has_no_path(self):
        made = []
        real = tempfile.mkdtemp
        def spy(*a, **k):
            made.append(real(*a, **k))
            return made[-1]
        def boom(ep):
            ep.state.place_order("r", "s", "A", 1)
            raise RuntimeError("x")
        with mock.patch("maf.state.tempfile.mkdtemp", spy):
            with mock.patch.dict(experiment.STRATEGIES, {"none": boom}):
                with self.assertRaises(RuntimeError):
                    run_episode(LLM, "mock", 0, None, "none", Config(), ledger="sqlite")
            self.assertEqual(len(made), 1)
            self.assertFalse(os.path.exists(made[0]))
            with tempfile.TemporaryDirectory() as d:
                ev = os.path.join(d, "e.jsonl")
                rec, _ = run_episode(LLM, "mock", 0, None, "selective", Config(), events_path=ev, ledger="sqlite")
                blob = json.dumps(rec) + open(ev, encoding="utf-8").read()
        self.assertFalse(os.path.exists(made[1]))
        self.assertNotIn(made[1], blob)
        self.assertNotIn("maf_ledger_", blob)
        self.assertEqual((rec["ledger_backend"], rec["ledger_io_s"] > 0), ("sqlite", True))

    def test_outcomes_match_memory(self):
        ignore = {"wall_time_s", "ledger_backend", "ledger_io_s", "tracking_overhead_s", "trace_overhead_s"}
        from maf.faults import FaultSpec
        for strat in ("none", "selective_gated", "full_restart"):
            for fault in (None, FaultSpec("malicious", "plan")):
                a = run_episode(LLM, "mock", 1, fault, strat, Config())[0]
                b = run_episode(LLM, "mock", 1, fault, strat, Config(), ledger="sqlite")[0]
                self.assertEqual({k: v for k, v in a.items() if k not in ignore},
                                 {k: v for k, v in b.items() if k not in ignore})
                self.assertEqual((a["ledger_backend"], a["ledger_io_s"]), ("memory", 0.0))

    def test_default_by_backend(self):
        ns = lambda **k: type("N", (), k)()
        self.assertEqual(experiment.resolve_ledger(ns(backend="llama")), "sqlite")
        self.assertEqual(experiment.resolve_ledger(ns(backend="mock")), "memory")
        self.assertEqual(experiment.resolve_ledger(ns(backend="mock", ledger="sqlite")), "sqlite")


if __name__ == "__main__":
    unittest.main()
