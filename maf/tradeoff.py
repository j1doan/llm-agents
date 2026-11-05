"""Cost-vs-safety chart (D27 item 1): per strategy cost, clean_success and irreversible damage, with Pareto marks.

  python -m maf.tradeoff results/x.jsonl [--by kind] [--x tokens|latency|calls] [--out-dir DIR]

Cost: mean prompt_tokens + completion_tokens, mean llm_calls, mean latency_s.
Damage: mean wrong_irreversible + duplicate_irreversible + unauthorized_final per episode.
Writes tradeoff.csv and tradeoff.svg (with --by kind: tradeoff_<kind>.svg, one per kind).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from maf.analyze import load

COST = {"tokens": lambda r: r["prompt_tokens"] + r["completion_tokens"],
        "latency": lambda r: r["latency_s"], "calls": lambda r: r["llm_calls"]}
XKEY = {"tokens": "mean_tokens", "latency": "mean_latency_s", "calls": "mean_calls"}
DAMAGE_FIELDS = ("wrong_irreversible", "duplicate_irreversible", "unauthorized_final")
COLUMNS = ["group", "strategy", "n", "mean_tokens", "mean_calls", "mean_latency_s", "clean_success",
           "task_success", "mean_damage", "pareto"]


def summarize(rows, by, x):
    """One dict per (group, strategy); group is the fault kind with by='kind', else 'all'."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["fault_kind"] if by == "kind" else "all", r["strategy"])].append(r)
    mean = lambda rs, f: statistics.fmean(f(r) for r in rs)
    out = [{"group": g, "strategy": s, "n": len(rs),
            "mean_tokens": mean(rs, COST["tokens"]), "mean_calls": mean(rs, COST["calls"]),
            "mean_latency_s": mean(rs, COST["latency"]),
            "clean_success": mean(rs, lambda r: float(r["clean_success"])),
            "task_success": mean(rs, lambda r: float(r["task_success"])),
            "mean_damage": mean(rs, lambda r: sum(r[k] for k in DAMAGE_FIELDS))}
           for (g, s), rs in sorted(groups.items())]
    mark_pareto(out, XKEY[x])
    return out


def mark_pareto(rows, xkey):
    """Within each group: pareto if no other row is <= cost and >= clean_success with one strict."""
    for a in rows:
        a["pareto"] = not any(
            b is not a and b["group"] == a["group"] and b[xkey] <= a[xkey]
            and b["clean_success"] >= a["clean_success"]
            and (b[xkey] < a[xkey] or b["clean_success"] > a["clean_success"]) for b in rows)


def svg_panel(rows, xkey, xlabel, title):
    w, h, ml, mr, mt, mb = 640, 420, 70, 40, 50, 60
    xs = [r[xkey] for r in rows]
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    px = lambda v: ml + (v - lo) / span * (w - ml - mr - 120)
    py = lambda v: mt + (1 - v) * (h - mt - mb)
    dmax = max(r["mean_damage"] for r in rows) or 1.0
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="sans-serif" font-size="12">',
         f'<title>{escape(title)}</title><rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{w / 2}" y="24" text-anchor="middle" font-size="14">{escape(title)}</text>',
         f'<line x1="{ml}" y1="{py(0)}" x2="{w - mr}" y2="{py(0)}" stroke="black"/>',
         f'<line x1="{ml}" y1="{py(1)}" x2="{ml}" y2="{py(0)}" stroke="black"/>',
         f'<text x="{w / 2}" y="{h - 14}" text-anchor="middle">{escape(xlabel)} '
         f'({lo:.4g} to {hi:.4g})</text>',
         f'<text transform="translate(18 {h / 2}) rotate(-90)" text-anchor="middle">clean_success rate</text>']
    for t in (0, 0.5, 1):
        o.append(f'<text x="{ml - 6}" y="{py(t) + 4}" text-anchor="end">{t}</text>')
    for r in rows:
        cx, cy, rad = px(r[xkey]), py(r["clean_success"]), 5 + 10 * r["mean_damage"] / dmax
        fill = "#1f77b4" if r["pareto"] else "white"
        o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad:.1f}" fill="{fill}" stroke="#1f77b4" '
                 f'stroke-width="2"/>')
        o.append(f'<text x="{cx + rad + 3:.1f}" y="{cy + 4:.1f}">{escape(r["strategy"])} '
                 f'(dmg {r["mean_damage"]:.2f})</text>')
    o.append(f'<text x="{ml}" y="{h - 36}" font-size="10">filled = Pareto-optimal; radius = mean irreversible '
             f'damage (wrong + duplicate + unauthorized_final)</text></svg>')
    return "\n".join(o)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("--by", choices=["kind"], help="one group (and one SVG) per fault kind")
    p.add_argument("--x", choices=list(COST), default="tokens")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args(argv)
    rows = load(args.results)
    with open(args.results, encoding="utf-8") as f:
        errors = sum(1 for line in f if line.strip()) - len(rows)
    summary = summarize(rows, args.by, args.x)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tradeoff.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(summary)
    mock = any(r.get("backend") == "mock" for r in rows)
    for g in sorted({r["group"] for r in summary}):
        name = "tradeoff.svg" if g == "all" else f"tradeoff_{g}.svg"
        title = f"Cost vs safety ({g})" + (" - MOCK - harness check only" if mock else "")
        panel = svg_panel([r for r in summary if r["group"] == g], XKEY[args.x], f"mean {args.x}", title)
        (out / name).write_text(panel, encoding="utf-8")
        print(f"wrote {out / name}")
    print(f"episodes={len(rows)} error_records_skipped={errors}")
    for r in summary:
        if r["pareto"]:
            print(f"pareto {r['group']}: {r['strategy']}")


if __name__ == "__main__":
    main()
