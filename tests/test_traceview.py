"""Trace viewer (D27): taint set, escaping, SVG validity, --list. Mock backend only."""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

from maf import traceview
from maf.faults import FaultSpec
from maf.llm import MockLLM
from maf.strategies import STRATEGIES
from maf.trace import EventTrace
from maf.workflow import Config, Episode
from maf.world import make_world


def run(fault, strategy):
    path = os.path.join(tempfile.mkdtemp(), "e.jsonl")
    key = f"mock|0|{fault.kind if fault else 'clean'}|{strategy}"
    ep = Episode(make_world(0), MockLLM(obey_prob=1.0), fault, Config(), EventTrace(path, key, 1))
    STRATEGIES[strategy](ep)
    ep.end_trace()
    return path, key


class TraceViewTest(unittest.TestCase):
    def events(self, fault, strategy):
        path, key = run(fault, strategy)
        return path, key, traceview.load_events(path, key)[1]

    def test_taint_equals_descendants(self):
        _, _, evs = self.events(FaultSpec("wrong_result", "research_pricing"), "selective_gated")
        fault = next(e for e in evs if e["kind"] == "fault")
        taint = traceview.taint_set(evs)
        desc, todo = {fault["id"]}, [fault["id"]]
        while todo:     # independent walk: children by parents, plus the altered artifact
            cur = todo.pop()
            for e in evs:
                if e["id"] not in desc and (cur in e["parents"] or (e["kind"] == "artifact" and e["payload"].get("fault_event") == cur)):
                    desc.add(e["id"])
                    todo.append(e["id"])
        self.assertEqual(taint, desc)
        self.assertGreater(len(taint), 3)

    def test_fault_edge_drawn_once_and_legacy_logs_still_dashed(self):
        _, _, evs = self.events(FaultSpec("wrong_result", "research_pricing"), "none")
        fault = next(e for e in evs if e["kind"] == "fault")
        shown = {e["id"] for e in evs}
        edges = [x for x in traceview._edges(evs, shown) if x[0] == fault["id"]]
        self.assertEqual(len(edges), len(set(edges)))
        self.assertTrue(edges and not any(dashed for *_, dashed in edges))      # real parent edge: solid
        for e in evs:       # a log from before T13: parent link absent, dashed edge and taint still there
            e["parents"] = [p for p in e["parents"] if p != fault["id"]]
        old = [x for x in traceview._edges(evs, shown) if x[0] == fault["id"]]
        self.assertTrue(any(dashed for *_, dashed in old))
        self.assertGreater(len(traceview.taint_set(evs)), 3)

    def test_render_svg_parses_and_has_banner(self):
        for fault, strat in [(None, "none"), (FaultSpec("wrong_result", "research_pricing"), "selective_gated"),
                             (FaultSpec("malicious", "notify"), "verify_hybrid")]:
            _, key, evs = self.events(fault, strat)
            page = traceview.render(key, 1, True, evs)
            svg = re.search(r"<svg xmlns.*?</svg>", page, re.S).group(0)
            ET.fromstring(svg)
            self.assertIn("MOCK - harness check only", page)

    def test_injected_script_escaped(self):
        _, key, evs = self.events(FaultSpec("wrong_result", "research_pricing"), "none")
        evs[0]["payload"]["text"] = "<script>alert(1)</script>"
        evs[2]["payload"]["content"] = {"x": "<script>alert(2)</script>", "p": "C:\\Users\\bob\\x.txt"}
        page = traceview.render(key, 1, True, evs)
        self.assertNotIn("<script>alert", page)
        self.assertNotIn("bob", page)
        self.assertEqual(page.count("<script>"), 1)     # only the viewer's own script

    def test_list_and_render_cli(self):
        path, key = run(FaultSpec("malicious", "notify"), "verify_hybrid")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            traceview.main([path, "--list"])
        self.assertIn(key, buf.getvalue())
        out = os.path.join(tempfile.mkdtemp(), "t.html")
        with contextlib.redirect_stdout(io.StringIO()):
            traceview.main([path, "--episode", key, "--out", out])
        with open(out, encoding="utf-8") as f:
            self.assertIn("rules", f.read())

    def test_default_attempt_is_latest_complete(self):
        path, key = run(None, "none")
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(x) for x in f]
        with open(path, "a", encoding="utf-8") as f:    # attempt 2 crashed: no episode_end
            f.write(json.dumps(dict(recs[0], attempt=2)) + "\n")
        self.assertEqual(traceview.load_events(path, key)[0], 1)
        self.assertEqual(traceview.load_events(path, key, 2)[2], False)


if __name__ == "__main__":
    unittest.main()
