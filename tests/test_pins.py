import json
import os
import socket
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest import mock

from maf import experiment, oracle, pins


class PinsTests(unittest.TestCase):
    def test_hash_and_size_of_small_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.gguf"
            f.write_bytes(b"abc" * 1000)
            p = pins.collect_pins(model_path=f)
        self.assertEqual(p["model_file"], {"file": "m.gguf", "size_bytes": 3000,
                                           "sha256": sha256(b"abc" * 1000).hexdigest()})

    def test_no_props_no_model_no_git(self):
        p = pins.collect_pins(settings={"temperature": 0})
        self.assertIsNone(p["llama_cpp_build"])
        self.assertIsNone(p["model_file"])
        self.assertEqual(p["model_settings"], {"temperature": 0})
        self.assertIn(p["git_dirty"], (None, True, False))
        self.assertEqual(p["packages"].keys(), {"tqdm", "pytest"})
        self.assertIsInstance(p["requirements_match"], bool)

    def test_props_build_info_only(self):
        p = pins.collect_pins(props={"build_info": "b10926-abc", "model_path": "C:/secret/x.gguf"})
        self.assertEqual(p["llama_cpp_build"], "b10926-abc")
        self.assertNotIn("secret", json.dumps(p))

    def test_requirements_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d) / "req.txt"
            r.write_text("tqdm==0.0.1\npytest==0.0.1\n")
            self.assertFalse(pins.collect_pins(requirements_path=r)["requirements_match"])

    def test_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.gguf"
            f.write_bytes(b"x")
            text = json.dumps(pins.collect_pins(model_path=f, props={"build_info": "b1"}))
        secrets = {os.path.expanduser("~"), socket.gethostname(), os.path.basename(os.path.expanduser("~"))}
        for s in secrets:
            self.assertNotIn(s, text)
            self.assertNotIn(s.replace("\\", "/"), text)
            self.assertNotIn(s.replace("\\", "\\\\"), text)

    def test_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "r.jsonl"
            a = pins.write_pins(res, {"python": "3.14"})
            same = pins.write_pins(res, {"python": "3.14"})
            b = pins.write_pins(res, {"python": "3.15"})
            c = pins.write_pins(res, {"python": "3.16"})
            self.assertEqual(a.name, "r.jsonl.pins.json")
            self.assertEqual(same, a)
            self.assertEqual((b.name, c.name), ("r.jsonl.pins.2.json", "r.jsonl.pins.3.json"))
            self.assertEqual(json.loads(a.read_text())["python"], "3.14")


GOOD = {"build_info": "b1", "model_path": "C:/secret/x.gguf", "default_generation_settings": {"n_ctx": 4096}}


class ServerContextTests(unittest.TestCase):
    """D34/D35: seed text in the pin file, and the llama n_ctx check before any episode."""

    def args(self, extra=()):
        return ["--backend", "llama", "--seeds", "0", "--kinds", "wrong_result", "--strategies", "none",
                "--log-dir", "", "--out", os.path.join(tempfile.mkdtemp(), "r.jsonl"), *extra]

    def refuses(self, props, entry, argv):
        boom = mock.Mock(side_effect=AssertionError("episode machinery reached"))
        with mock.patch.object(experiment, "fetch_props", return_value=props),                 mock.patch.object(experiment, "make_llm", boom), mock.patch.object(oracle, "make_llm", boom),                 mock.patch.object(experiment, "run_episode", boom):
            with self.assertRaises(SystemExit) as cm:
                entry(argv)
        self.assertIn("refusing to start", str(cm.exception))
        boom.assert_not_called()
        return cm.exception

    def test_n_ctx_lookup_accepts_both_places(self):
        self.assertEqual(pins.server_n_ctx(GOOD), 4096)
        self.assertEqual(pins.server_n_ctx({"n_ctx": 8192}), 8192)
        self.assertIsNone(pins.server_n_ctx({"error": "x"}))
        self.assertIsNone(pins.server_n_ctx(None))

    def test_pin_stores_n_ctx_not_paths(self):
        text = json.dumps(pins.collect_pins(props=GOOD))
        self.assertEqual(json.loads(text)["server_n_ctx"], 4096)
        self.assertNotIn("secret", text)

    def test_right_n_ctx_passes_and_seed_text_matches(self):
        ns = mock.Mock(backend="llama", base_url="u", temperature=0.0, max_tokens=512, seed_offset=7, ledger=None,
                       out=os.path.join(tempfile.mkdtemp(), "r.jsonl"))
        with mock.patch.object(experiment, "fetch_props", return_value=GOOD):
            out = experiment.pin_run(ns, "g")
        got = json.loads(out.read_text())
        self.assertEqual(got["server_n_ctx"], 4096)
        self.assertEqual(got["model_settings"]["seed_policy"], "fixed seed (--seed-offset 7)")

    def test_wrong_or_missing_n_ctx_refuses_before_any_episode(self):
        wrong = {"build_info": "b1", "default_generation_settings": {"n_ctx": 8192}}
        for props in (wrong, {"build_info": "b1"}, {"error": "unreachable"}):
            self.refuses(props, experiment.main, self.args())
            self.refuses(props, oracle.main, ["probe", *self.args()[:2], "--seeds", "0", "--out",
                                              os.path.join(tempfile.mkdtemp(), "p.jsonl"), "--log-dir", ""])
        exc = self.refuses(wrong, experiment.main, self.args())
        self.assertIn("8192", str(exc))

    def test_mock_skips_the_check(self):
        out = os.path.join(tempfile.mkdtemp(), "r.jsonl")
        boom = mock.Mock(side_effect=AssertionError("props fetched"))
        with mock.patch.object(experiment, "fetch_props", boom):
            experiment.main(["--backend", "mock", "--seeds", "0", "--kinds", "wrong_result", "--strategies", "none",
                             "--log-dir", "", "--out", out, "--seed-offset", "3"])
            oracle.main(["probe", "--backend", "mock", "--seeds", "0", "--out", out + "p", "--log-dir", ""])
        boom.assert_not_called()
        for f in (out, out + "p"):
            pin = json.loads(Path(f + ".pins.json").read_text())
            self.assertIsNone(pin["server_n_ctx"])
        self.assertEqual(json.loads(Path(out + ".pins.json").read_text())["model_settings"]["seed_policy"],
                         "fixed seed (--seed-offset 3)")


if __name__ == "__main__":
    unittest.main()
