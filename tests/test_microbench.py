"""Run with:  python -m unittest discover -s tests -t .  (from the project root)"""
import os
import tempfile
import unittest

from maf import microbench
from maf.microbench import make_runs, record


def closure(shape, n, idx, track_state):
    tr, reads = make_runs(shape, n)
    record(tr, reads)
    return tr.downstream(f"r{idx}", track_state)


def reference_downstream(tr, run_id, track_state):
    """The pre-D33 algorithm: sort with a list scan per element."""
    closure = {run_id}
    live = [tr.runs[r] for r in tr.current.values()]
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
    return sorted(closure, key=lambda rid: tr.order.index(tr.runs[rid].stage))


class MicrobenchTests(unittest.TestCase):
    def test_chain_closure_sizes(self):
        for n in (1, 5, 40):
            self.assertEqual(len(closure("chain", n, 0, True)), n)
            self.assertEqual(len(closure("chain", n, n - 1, True)), 1)
            self.assertEqual(len(closure("chain", n, n // 2, False)), n - n // 2)

    def test_dag_uses_both_edge_kinds(self):
        n = 30
        with_state = closure("dag", n, 0, True)
        msg_only = closure("dag", n, 0, False)
        self.assertLess(len(msg_only), len(with_state))
        self.assertEqual(len(closure("dag", n, n - 1, True)), 1)
        self.assertEqual(with_state, sorted(with_state, key=lambda r: int(r[1:])))

    def test_downstream_matches_reference(self):
        for shape in ("chain", "dag"):
            for n in (1, 2, 5, 17, 60, 200):
                tr, reads = make_runs(shape, n)
                record(tr, reads)
                for idx in sorted({0, n // 3, n // 2, n - 1}):
                    for ts in (True, False):
                        self.assertEqual(tr.downstream(f"r{idx}", ts),
                                         reference_downstream(tr, f"r{idx}", ts))

    def test_cli_runs_with_tiny_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "mb.csv")
            rows = microbench.main(["--sizes", "5,8", "--repeats", "2", "--out", out])
            self.assertEqual(len(rows), 2 * 2 * 7)
            with open(out) as f:
                self.assertEqual(len(f.readlines()), len(rows) + 1)

    def test_trace_section_on_off(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.csv")
            on, off = microbench.main(["--trace", "--seeds", "0", "--repeats", "1", "--out", out])
            self.assertEqual(on[1], off[1])
            self.assertEqual(on[1], 6 * len(microbench.TRACE_STRATEGIES))
            self.assertGreater(on[2], 0)
            self.assertGreater(on[3], 0)
            self.assertEqual((off[2], off[3], off[4], off[5], off[6]), (0, 0, 0, 0, 0))
            self.assertEqual(os.listdir(d), ["t.csv"])


if __name__ == "__main__":
    unittest.main()
