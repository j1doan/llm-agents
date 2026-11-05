"""Run logging: a full DEBUG file log per run plus a tqdm-safe console log.

Modules use `log = logging.getLogger(__name__)`; entry points call `setup_logging()` once.
Levels: DEBUG prompts, raw model text and effects; INFO stage runs, LLM calls, recovery steps;
WARNING anything that smells (invalid JSON, truncated output, injected faults, irreversible effects).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from tqdm import tqdm

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"   # project root, independent of the cwd
FILE_FMT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def run_dir_for(log_dir, kind: str, label: str = "") -> Path:
    name = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}" + (f"_{label}" if label else "")
    path = (Path(log_dir) / name).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(run_dir: Path | None = None, console_level: str = "WARNING") -> None:
    root = logging.getLogger("maf")
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False
    console = TqdmHandler()
    console.setLevel(getattr(logging, console_level.upper()))
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    if run_dir is not None:
        fh = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FILE_FMT, datefmt="%H:%M:%S"))
        root.addHandler(fh)


def clip(text, n: int = 300) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + f"...[+{len(text) - n} chars]"
