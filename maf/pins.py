"""Pin file (D13): versions and hashes that tie a results file to the exact software and model used.

CLI:  python -m maf.pins [--model PATH] [--server URL]
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("tqdm", "pytest")


def fetch_props(base_url, timeout=5):
    """llama-server /props as a dict, or {"error": ...} if unreachable."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/props", timeout=timeout) as r:
            return json.load(r)
    except Exception as e:
        return {"error": repr(e)}


def server_n_ctx(props):
    """Context size from /props. llama-server nests it as default_generation_settings.n_ctx (per slot);
    a top-level n_ctx is accepted too. None if absent."""
    props = props or {}
    return (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")


def require_server_ctx(props, expected):
    """D35: refuse to start when the server's context is missing or not the intended one."""
    got = server_n_ctx(props)
    if got != expected:
        raise SystemExit(f"refusing to start: llama-server /props reports n_ctx={got!r}, expected {expected} "
                         f"(D15/D35). Restart the server with -c {expected} (and --parallel 1), or check --base-url.")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _requirements(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name, sep, ver = line.strip().partition("==")
        if sep:
            pins[name.lower()] = ver
    return pins


def collect_pins(backend="mock", model_label="mock", settings=None, model_path=None, props=None,
                 requirements_path=ROOT / "REQUIREMENTS.txt"):
    """settings: D15 model settings (temperature, context, max output tokens, seed policy), stored as given."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        wanted = _requirements(requirements_path)
    except OSError:
        wanted = None
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    model = None
    if model_path:
        model = {"file": Path(model_path).name, "size_bytes": Path(model_path).stat().st_size,
                 "sha256": sha256_file(model_path)}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "packages": versions,
        "requirements_match": wanted is not None and all(versions[n] == wanted.get(n) for n in PACKAGES),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "llama_cpp_build": (props or {}).get("build_info"),
        "server_n_ctx": server_n_ctx(props),
        "model_file": model,
        "model_settings": settings,
        "backend": backend,
        "model_label": model_label,
    }


def write_pins(results_path, pins):
    """Write <results>.pins.json; never overwrite a different file, use .pins.2.json, .pins.3.json, ..."""
    results_path = Path(results_path)
    text = json.dumps(pins, indent=2) + "\n"
    n = 1
    while True:
        out = results_path.with_name(results_path.name + (".pins.json" if n == 1 else f".pins.{n}.json"))
        if not out.exists():
            out.write_text(text, encoding="utf-8")
            return out
        old = json.loads(out.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "timestamp_utc"} == {k: v for k, v in pins.items() if k != "timestamp_utc"}:
            return out
        n += 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Print run pins.")
    p.add_argument("--model", help="model file to hash (slow for ~7 GB)")
    p.add_argument("--server", help="llama-server base URL, to read build info from /props")
    args = p.parse_args(argv)
    props = fetch_props(args.server) if args.server else None
    pins = collect_pins("llama" if args.server else "mock", "cli", model_path=args.model, props=props)
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
