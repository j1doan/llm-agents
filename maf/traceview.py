"""Trace viewer (D27 item 2): one episode's event dependency graph as a standalone HTML file (inline SVG, CSS, JS).

  python -m maf.traceview <events.jsonl> --list
  python -m maf.traceview <events.jsonl> --episode "model|seed|fault|strategy" [--attempt N] [--out FILE.html]

Nodes are events (message and state_read events are folded into their run's node), edges are `parents`.
The fault's taint is every event reachable from it through parent edges, plus the artifact it altered
(payload.fault_event), which logs written before T13 do not list as a parent of the fault.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import Counter

from maf.trace import load_attempts

STAGES = ["research_pricing", "research_policy", "plan", "execute", "notify"]
FOLDED = {"message", "state_read"}
SKIP = FOLDED | {"episode_end"}
W, H, GAP, COLW, LEFT, TOP = 176, 38, 8, 216, 90, 54


def load_events(path, key, attempt=None):
    """(attempt, events, complete) for one episode; default attempt = latest complete, else latest."""
    with open(path, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, f) if r["episode"] == key]
    if not recs:
        raise SystemExit(f"episode not found: {key}")
    if attempt is None:
        done = {r["attempt"] for r in recs if r["kind"] == "episode_end"}
        attempt = max(done or {r["attempt"] for r in recs})
    evs = [r for r in recs if r["attempt"] == attempt]
    if not evs:
        raise SystemExit(f"no attempt {attempt} for {key}")
    return attempt, evs, any(r["kind"] == "episode_end" for r in evs)


def taint_set(events):
    """Ids of the fault events and everything downstream of them."""
    kids = {}
    for e in events:
        for p in e["parents"]:
            kids.setdefault(p, []).append(e["id"])
        fe = (e["payload"] or {}).get("fault_event")
        if e["kind"] == "artifact" and fe:
            kids.setdefault(fe, []).append(e["id"])
    seen, todo = set(), [e["id"] for e in events if e["kind"] == "fault"]
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(kids.get(i, []))
    return seen


def _stage(e):
    name = e["stage"] or (e["run_id"] or "").split("#")[0]
    return STAGES.index(name) if name in STAGES else None


def _band(e):
    try:
        return int(e["run_id"].split("#")[1]) - 1
    except (AttributeError, IndexError, ValueError):
        return None


def _scrub(s):
    """No absolute paths in the output (privacy)."""
    s = re.sub(r"[A-Za-z]:[\\/][^\s\"',]*", "<path>", s)
    return re.sub(r"(?<![\w.])/(?:Users|home|tmp|var|mnt)/[^\s\"',]*", "<path>", s)


def _label(e):
    p, k = e["payload"] or {}, e["kind"]
    if k == "fault":
        return p.get("fault", "?") + (" (noop)" if p.get("noop") else "")
    if k == "verify":
        return f"{p.get('verdict', '?')} [{p.get('checker', '?')}]"
    if k == "tool_action":
        return f"{p.get('tool', '?')} {p.get('status', '?')}"
    if k == "state_write":
        return str(p.get("key", ""))
    if k == "rollback":
        return f"{len(p.get('runs', []))} runs, {p.get('effects', 0)} effects"
    return "recovery" if p.get("recovery") else ""


def _cls(e):
    p, k = e["payload"] or {}, e["kind"]
    sub = {"verify": p.get("verdict"), "tool_action": p.get("status")}.get(k, "")
    return f"n k-{k}" + (f" s-{sub}" if sub else "")


def layout(events):
    """node id -> (x, y) from stage column and run band; nodes in one cell stack in event order."""
    nodes = [e for e in events if e["kind"] not in SKIP]
    cell, pos_of = {}, {}
    byid = {e["id"]: e for e in events}
    for e in nodes:
        col, band = _stage(e), _band(e)
        if col is None or band is None:     # rollback: the latest column and band among its parents
            ps = [byid[i] for i in e["parents"] if i in byid]
            col = max([_stage(q) for q in ps if _stage(q) is not None] or [0])
            band = max([_band(q) for q in ps if _band(q) is not None] or [0])
        cell.setdefault((col, band), []).append(e["id"])
    bands = sorted({b for _, b in cell})
    top, y0 = {}, TOP
    for b in bands:
        top[b] = y0
        y0 += max(len(v) for (c, bb), v in cell.items() if bb == b) * (H + GAP) + 24
    for (c, b), ids in cell.items():
        for i, nid in enumerate(ids):
            pos_of[nid] = (LEFT + c * COLW, top[b] + i * (H + GAP))
    return nodes, pos_of, top, y0


def _edges(events, shown):
    out = []
    for e in events:
        if e["id"] not in shown:
            continue
        for p in e["parents"]:
            if p in shown:
                out.append((p, e["id"], False))
        fe = (e["payload"] or {}).get("fault_event")
        if e["kind"] == "artifact" and fe in shown and fe not in e["parents"]:   # old logs: no parent link
            out.append((fe, e["id"], True))
    return out


def _detail(e, taint, folded):
    head = [f"{e['id']}  {e['kind']}", f"stage {e['stage'] or '-'}   run {e['run_id'] or '-'}   t {e['t']}",
            f"parents {', '.join(e['parents']) or '-'}", f"tainted by fault: {'YES' if e['id'] in taint else 'no'}",
            "", json.dumps(e["payload"], indent=1, ensure_ascii=False, default=str)]
    for f in folded:
        head += ["", f"-- folded {f['id']} {f['kind']} --", json.dumps(f["payload"], indent=1, ensure_ascii=False, default=str)]
    return _scrub("\n".join(head))


def summary(key, attempt, complete, events, taint):
    end = next((e for e in events if e["kind"] == "episode_end"), None)
    ver = Counter(f"{e['payload'].get('verdict')}/{e['payload'].get('checker')}" for e in events if e["kind"] == "verify")
    tools = Counter(e["payload"].get("status") for e in events if e["kind"] == "tool_action")
    faults = [f"{e['payload'].get('fault')}{' (noop)' if e['payload'].get('noop') else ''}" for e in events if e["kind"] == "fault"]
    return [("attempt", f"{attempt}" + ("" if complete else " (incomplete, no episode_end)")),
            ("events", len(events)), ("fault", ", ".join(faults) or "none"),
            ("tainted events", len(taint)), ("verify", ", ".join(f"{k} x{v}" for k, v in sorted(ver.items())) or "-"),
            ("tool actions", ", ".join(f"{k} x{v}" for k, v in sorted(tools.items())) or "-"),
            ("rollbacks", sum(e["kind"] == "rollback" for e in events)),
            ("aborted", end["payload"].get("aborted") if end else "?"),
            ("sim time s", max(e["t"] for e in events))]


CSS = """
:root{color-scheme:light dark;--bg:#fff;--tx:#1a1a1a;--mute:#555;--bd:#555;--edge:#8a8f98;--red:#c0182b;
--art:#e6e9ee;--sw:#d9dbf2;--fault:#f5b5b5;--acc:#c3ecc9;--rej:#ffd0a8;--applied:#bcd8f7;--held:#fbe39a;
--released:#b4ebe3;--dropped:#e0e0e0;--skipped:#f0f0f0;--roll:#e1c8f2;--panel:#f6f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--tx:#ececec;--mute:#aaa;--bd:#9aa0aa;--edge:#7b818c;--red:#ff6b7a;
--art:#2c3038;--sw:#2e3057;--fault:#7a2030;--acc:#1f5a2c;--rej:#8a4a12;--applied:#1f4a7d;--held:#7d6410;
--released:#12665c;--dropped:#3a3a3a;--skipped:#2a2a2a;--roll:#5a2f78;--panel:#1d2025}}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--tx);font:14px/1.4 system-ui,Segoe UI,sans-serif}
h1{font-size:17px;margin:0 0 4px;word-break:break-all}
.mock{display:inline-block;background:#b00020;color:#fff;font-weight:700;padding:2px 8px;border-radius:3px;margin:4px 0}
.sum{display:flex;flex-wrap:wrap;gap:4px 18px;color:var(--mute);margin:4px 0 8px}.sum b{color:var(--tx)}
.lg{display:inline-flex;align-items:center;gap:4px;margin:2px 10px 2px 0;font-size:12px;color:var(--mute)}
.wrap{overflow:auto;border:1px solid var(--edge);background:var(--panel)}
svg text{fill:var(--tx);font:11px system-ui,sans-serif;pointer-events:none}svg .l2{fill:var(--mute)}
svg .hd{font-weight:700;font-size:12px}
.n{stroke:var(--bd);stroke-width:1.2;fill:var(--art);cursor:pointer}
.k-state_write{fill:var(--sw)}.k-fault{fill:var(--fault);stroke:var(--red);stroke-width:2.5}
.k-verify.s-accept{fill:var(--acc)}.k-verify.s-reject{fill:var(--rej)}.k-rollback{fill:var(--roll)}
.k-tool_action.s-applied{fill:var(--applied)}.k-tool_action.s-held{fill:var(--held)}
.k-tool_action.s-released{fill:var(--released)}
.k-tool_action.s-dropped{fill:var(--dropped);stroke-dasharray:4 3}
.k-tool_action.s-skipped{fill:var(--skipped);stroke-dasharray:1 3}
.n.taint{stroke:var(--red);stroke-width:3}.n.sel{stroke-width:4;stroke:var(--tx)}
.tb{fill:var(--red)}.e{fill:none;stroke:var(--edge);stroke-width:1.2}
.e.taint{stroke:var(--red);stroke-width:1.8}.e.impl{stroke-dasharray:5 3}
.ah{fill:var(--edge)}.ah.taint{fill:var(--red)}
pre{background:var(--panel);border:1px solid var(--edge);padding:8px;margin:8px 0 0;max-height:320px;overflow:auto;
white-space:pre-wrap;word-break:break-word;font:12px/1.35 ui-monospace,Consolas,monospace}
"""

JS = """const nodes=document.querySelectorAll('.n[data-id]'),out=document.getElementById('out');let pin=null;
function show(id){out.textContent=D[id]}
nodes.forEach(n=>{n.addEventListener('mouseover',()=>{if(!pin)show(n.dataset.id)});
n.addEventListener('click',()=>{nodes.forEach(m=>m.classList.remove('sel'));
if(pin===n.dataset.id){pin=null;return}pin=n.dataset.id;n.classList.add('sel');show(pin)})});"""

LEGEND = [("k-fault", "fault"), ("k-artifact", "artifact"), ("k-state_write", "state_write"),
          ("k-verify s-accept", "verify accept"), ("k-verify s-reject", "verify reject"),
          ("k-tool_action s-applied", "tool applied"), ("k-tool_action s-held", "tool held"),
          ("k-tool_action s-released", "tool released"), ("k-tool_action s-dropped", "tool dropped"),
          ("k-tool_action s-skipped", "tool skipped"), ("k-rollback", "rollback"),
          ("k-artifact taint", "red border = tainted by the fault")]


def render(key, attempt, complete, events):
    taint = taint_set(events)
    nodes, pos, top, height = layout(events)
    shown = set(pos)
    esc = lambda s: html.escape(_scrub(str(s)))  # noqa: E731
    home = {}                                    # run_id -> node that holds its folded message/state_read events
    for e in nodes:
        if e["run_id"] and (e["run_id"] not in home or e["kind"] == "artifact"):
            home[e["run_id"]] = e["id"]
    fold = {}
    for e in events:
        if e["kind"] in FOLDED and e["run_id"] in home:
            fold.setdefault(home[e["run_id"]], []).append(e)
    details = {e["id"]: _detail(e, taint, fold.get(e["id"], [])) for e in nodes}

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{LEFT + COLW * len(STAGES)}" height="{height}" '
           f'viewBox="0 0 {LEFT + COLW * len(STAGES)} {height}">',
           '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
           '<path class="ah" d="M0,0L8,4L0,8z"/></marker><marker id="t" markerWidth="8" markerHeight="8" refX="7" '
           'refY="4" orient="auto"><path class="ah taint" d="M0,0L8,4L0,8z"/></marker></defs>']
    for c, s in enumerate(STAGES):
        svg.append(f'<text class="hd" x="{LEFT + c * COLW}" y="20">{s}</text>')
    for b, y in top.items():
        svg.append(f'<text class="l2" x="6" y="{y + 14}">{"first pass" if b == 0 else f"rerun {b}"}</text>')
    for p, c, impl in _edges(events, shown):
        (px, py), (cx, cy) = pos[p], pos[c]
        t = p in taint and c in taint
        if px < cx:
            x1, y1, x2, y2 = px + W, py + H / 2, cx, cy + H / 2
            d = f"M{x1},{y1} C{x1 + 40},{y1} {x2 - 40},{y2} {x2},{y2}"
        elif px == cx:
            x1, y1, y2 = px, py + H / 2, cy + H / 2
            d = f"M{x1},{y1} C{x1 - 36},{y1} {x1 - 36},{y2} {x1},{y2}"
        else:
            x1, y1, x2, y2 = px, py + H / 2, cx + W, cy + H / 2
            d = f"M{x1},{y1} C{x1 - 40},{y1} {x2 + 40},{y2} {x2},{y2}"
        svg.append(f'<path class="e{" taint" if t else ""}{" impl" if impl else ""}" d="{d}" '
                   f'marker-end="url(#{"t" if t else "a"})"/>')
    for e in nodes:
        x, y = pos[e["id"]]
        cls = _cls(e) + (" taint" if e["id"] in taint else "")
        svg.append(f'<g><rect class="{cls}" data-id="{esc(e["id"])}" x="{x}" y="{y}" width="{W}" height="{H}" rx="5"/>'
                   f'<text x="{x + 6}" y="{y + 15}">{esc(e["id"])} {esc(e["kind"])}</text>'
                   f'<text class="l2" x="{x + 6}" y="{y + 29}">{esc(_label(e)[:30])}</text>'
                   + (f'<circle class="tb" cx="{x + W - 8}" cy="{y + 8}" r="4"/>' if e["id"] in taint else "") + "</g>")
    svg.append("</svg>")

    legend = "".join(f'<span class="lg"><svg width="22" height="14"><rect class="{c}" x="1" y="1" width="20" height="12" '
                     f'rx="3"/></svg>{esc(t)}</span>' for c, t in LEGEND)
    summ = "".join(f"<span>{esc(k)}: <b>{esc(v)}</b></span>" for k, v in summary(key, attempt, complete, events, taint))
    mock = '<div class="mock">MOCK - harness check only</div>' if key.split("|")[0] == "mock" else ""
    data = json.dumps(details).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>trace {esc(key)}</title>'
            f'<style>{CSS}</style></head><body><h1>{esc(key)}</h1>{mock}<div class="sum">{summ}</div>'
            f'<div>{legend}</div><div class="wrap">{"".join(svg)}</div>'
            f'<pre id="out">Hover or click a node to see its payload.</pre><script>const D={data};{JS}</script></body></html>')


def main(argv=None):
    p = argparse.ArgumentParser(description="Render one episode's trace as a standalone HTML dependency graph.")
    p.add_argument("events")
    p.add_argument("--episode", help="model|seed|fault|strategy")
    p.add_argument("--attempt", type=int, help="default: latest complete attempt")
    p.add_argument("--out", help="default: <episode key, filesystem-safe>.html")
    p.add_argument("--list", action="store_true", help="print the episode keys in the file")
    a = p.parse_args(argv)
    if a.list:
        for k, n in sorted(load_attempts(a.events).items()):
            print(f"{k}\tattempts={n}")
        return 0
    if not a.episode:
        p.error("--episode is required unless --list")
    attempt, events, complete = load_events(a.events, a.episode, a.attempt)
    out = a.out or re.sub(r"[^\w.@-]+", "_", a.episode) + ".html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(a.episode, attempt, complete, events))
    print(f"wrote {os.path.basename(out)} ({len(events)} events, attempt {attempt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
