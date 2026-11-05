"""Run with:  python -m unittest discover -s tests -t .  (from the project root)"""
import contextlib
import csv
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from maf.tradeoff import COLUMNS, main, mark_pareto


def rec(strategy, clean, tokens, kind="wrong_result", wrong=0):
    return {"strategy": strategy, "fault_kind": kind, "backend": "mock", "clean_success": clean,
            "task_success": clean, "prompt_tokens": tokens, "completion_tokens": 0, "llm_calls": 5,
            "latency_s": 1.0, "wrong_irreversible": wrong, "duplicate_irreversible": 0,
            "unauthorized_final": 0}


def write(tmp, records, n_errors=0):
    lines = [json.dumps(r) for r in records] + [json.dumps({"key": "k", "error": "boom"})] * n_errors
    path = Path(tmp) / "r.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TradeoffTests(unittest.TestCase):
    def test_pareto_marking(self):
        rows = [{"group": "g", "k": c, "clean_success": s} for c, s in [(1, 0.5), (2, 1.0), (3, 1.0), (2, 0.4)]]
        mark_pareto(rows, "k")
        self.assertEqual([r["pareto"] for r in rows], [True, True, False, False])

    def test_csv_and_svg(self):
        records = [rec("none", False, 100, wrong=1), rec("selective", True, 300), rec("full_restart", True, 500)]
        with tempfile.TemporaryDirectory() as tmp:
            main([write(tmp, records), "--out-dir", tmp])
            with open(Path(tmp) / "tradeoff.csv", encoding="utf-8") as f:
                table = list(csv.DictReader(f))
            self.assertEqual(list(table[0]), COLUMNS)
            self.assertEqual({r["strategy"]: r["pareto"] for r in table},
                             {"none": "True", "selective": "True", "full_restart": "False"})
            svg = ET.parse(Path(tmp) / "tradeoff.svg").getroot()
            self.assertIn("MOCK - harness check only", svg.findtext("{http://www.w3.org/2000/svg}title"))

    def test_error_records_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main([write(tmp, [rec("none", True, 1)], n_errors=2), "--out-dir", tmp, "--by", "kind"])
            self.assertIn("episodes=1 error_records_skipped=2", buf.getvalue())
            ET.parse(Path(tmp) / "tradeoff_wrong_result.svg")


if __name__ == "__main__":
    unittest.main()
