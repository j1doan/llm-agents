"""Write CSV copies of the data tables in report/PHASE1_DUMP.md to results/cleaned_data/.

Values are copied from the dump, not recomputed from raw results. The numbers that the dump
gives only in sentences are listed in PROSE_TABLES below, each with its dump section.

    python scripts/clean_dump_tables.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "report" / "PHASE1_DUMP.md"
OUT = ROOT / "results" / "cleaned_data"


def clean(cell):
    cell = re.sub(r"\*\*(.*?)\*\*", r"\1", cell.strip())
    return cell.replace("`", "")


def split_row(line):
    line = line.strip().removeprefix("|").removesuffix("|")
    return [clean(c) for c in re.split(r"(?<!\\)\|", line)]


def is_separator(line):
    return line.strip().startswith(("|-", "| -", "|:", "| :")) and set(line.strip()) <= set("|-: ")


def parse_tables(path):
    """Markdown tables in the dump, each with the heading it sits under."""
    lines = path.read_text(encoding="utf-8").splitlines()
    tables, heading, i = [], "", 0
    while i < len(lines):
        m = re.match(r"^#{2,4}\s+(.*)$", lines[i])
        if m:
            heading = m.group(1).strip()
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            tables.append({"heading": heading, "header": split_row(lines[i]), "rows": rows})
            i = j
            continue
        i += 1
    return tables


def snake(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    print(path.relative_to(OUT), len(rows), "rows")


PROSE_TABLES = {
    "12_verifier_detection.csv": (
        ["strategy", "tp", "fn", "fp", "tn", "recall", "precision", "dump_section"],
        [["verify_all", 118, 12, 61, 905, 0.91, 0.66, "4.3"],
         ["verify_hybrid", 129, 1, 65, 914, 0.99, 0.66, "4.5"]]),
    "13_grid_run_summary.csv": (
        ["item", "value", "dump_section"],
        [["episodes", 1520, "3.2"], ["strategies", 8, "3.2"], ["seeds", 10, "3.2"],
         ["conditions", 19, "3.2"], ["error_records", 0, "3.2"], ["episodes_per_strategy", 190, "3.2"],
         ["wall_time_hours_sum_of_episodes", 9.79, "3.2"],
         ["cross_run_episodes_compared", 570, "3.3"], ["cross_run_identical_outcomes", 570, "3.3"],
         ["first_attempt_repeat_episodes", 285, "3.3"], ["first_attempt_repeat_identical", 285, "3.3"]]),
    "14_trace_event_counts.csv": (
        ["event_kind", "count"],
        [["message", 12440], ["artifact", 9970], ["tool_action", 7020], ["state_read", 6435],
         ["state_write", 5730], ["verify", 3274], ["fault", 1440], ["rollback", 750],
         ["episode_end", 1520]]),
    "15_overhead.csv": (
        ["measure", "value", "unit", "source", "dump_section"],
        [["trace_file_size", 21764574, "bytes", "grid", "5.8"],
         ["trace_events_per_episode_mean", 31.96, "events", "grid", "5.8"],
         ["trace_overhead_per_episode_mean", 1.09, "ms", "grid", "5.8"],
         ["trace_overhead_per_episode_max", 10.9, "ms", "grid", "5.8"],
         ["tracking_overhead_per_episode_mean", 3.3, "us", "grid", "5.9"],
         ["tracking_overhead_per_episode_max", 29.7, "us", "grid", "5.9"],
         ["record_read_cost", "0.12-0.15", "us", "microbench", "5.9"],
         ["trace_overhead_per_episode_mean", 0.46, "ms", "mock episodes", "5.9"],
         ["trace_overhead_per_event", 14, "us", "mock episodes", "5.9"],
         ["tracker_per_episode_mock", 0.9, "us", "mock episodes", "5.9"],
         ["ledger_io_per_episode_median", 0.088, "s", "grid", "5.11"],
         ["ledger_io_per_episode_max", 0.71, "s", "grid", "5.11"]]),
}


def main():
    tables = parse_tables(DUMP)

    def find(section, first_cell, ncols):
        for t in tables:
            if t["heading"].startswith(section) and t["header"][0] == first_cell and len(t["header"]) == ncols:
                return t
        raise KeyError((section, first_cell, ncols))

    def plain(name, t):
        write(OUT / name, [snake(h) for h in t["header"]], t["rows"])

    plain("01_none_propagation.csv", find("4.1", "Fault", 6))
    plain("02_full_restart.csv", find("4.2", "Fault", 8))
    plain("03_verify_all.csv", find("4.3", "Fault", 7))

    gate = find("4.4", "Fault", 5)  # "wrong_result (n=50)" -> fault, n_episodes
    rows = []
    for r in gate["rows"]:
        m = re.match(r"(\w+) \(n=(\d+)\)", r[0])
        rows.append([m.group(1), m.group(2)] + r[1:])
    write(OUT / "04_gate_clean_success.csv",
          ["fault", "n_episodes", "full_restart", "full_restart_gated", "selective", "selective_gated"], rows)
    plain("05_gate_effect_counts.csv", find("4.4", "Strategy", 5))
    plain("06_verify_hybrid.csv", find("4.5", "Fault", 5))
    plain("07_selective.csv", find("4.6", "Fault", 7))
    plain("08_oracle_tracker.csv", find("4.6", "Tracker", 9))

    probes = find("4.6", "Probe", 5)  # "0.13 / 0.00" -> two columns; "n/a" -> empty
    rows = []
    for r in probes["rows"]:
        strict, decision = (x.strip().replace("n/a", "") for x in r[4].split(" / ", 1))
        rows.append(r[:4] + [strict, decision])
    write(OUT / "09_oracle_probes.csv",
          ["probe", "tracker_pred_selective", "strict_true", "decision_true",
           "precision_strict", "precision_decision"], rows)

    tradeoff = find("4.8", "Strategy", 5)  # "0.00 / 1833 *" -> long form
    rows = []
    for r in tradeoff["rows"]:
        for fault, cell in zip(tradeoff["header"][1:], r[1:]):
            on_front = cell.endswith("*")
            success, tokens = (x.strip() for x in cell.rstrip("* ").split("/"))
            rows.append([r[0], fault, success, tokens, str(on_front)])
    write(OUT / "10_tradeoff_long.csv",
          ["strategy", "fault", "clean_success", "mean_tokens", "pareto_front"], rows)

    write(OUT / "11_microbench_closure_chain.csv",
          ["n_stages", "before_d33_median_us", "after_d33_median_us"], find("5.9", "N", 3)["rows"])

    for name, (header, rows) in PROSE_TABLES.items():
        write(OUT / name, header, rows)

    for name, t in [("stages.csv", find("1.2", "Stage", 4)), ("fault_kinds.csv", find("1.4", "Kind", 4)),
                    ("strategies.csv", find("1.5", "Strategy", 3)), ("metrics_fields.csv", find("3.3", "Field", 2))]:
        plain("definitions/" + name, t)


if __name__ == "__main__":
    main()
