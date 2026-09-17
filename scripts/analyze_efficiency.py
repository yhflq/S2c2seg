#!/usr/bin/env python3
"""Aggregate efficiency benchmark results into a single report.

Collects the vocabulary-scaling sweep, FLOPs accounting, and end-to-end runs.
Robust to missing inputs: sections whose data files are absent are skipped
with a note.
"""
import argparse
import csv
import os.path as osp
import re
import sys
from datetime import datetime

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

OUT = "results/efficiency"
BLUE, ORANGE, GREEN, GRAY = "#2563eb", "#ea580c", "#059669", "#6b7280"

E2E_META = {
    # name: (protocol, variant, is_baseline)
    "voc21_baseline": ("voc21", "ProxyCLIP baseline", True),
    "voc21_r1": ("voc21", "+S2C2Seg++ R1 (default)", False),
    "ctx59_baseline": ("ctx59", "ProxyCLIP baseline", True),
    "ctx59_r1": ("ctx59", "+S2C2Seg++ R1 (default)", False),
    "stuff_baseline": ("stuff171", "ProxyCLIP baseline", True),
    "stuff_r1": ("stuff171", "+S2C2Seg++ R1 (default)", False),
}

ITER_RE = re.compile(
    r"(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2}) - mmengine - INFO - "
    r"Iter\(test\) \[ *(\d+)/(\d+)\]")


def read_csv(path):
    if not osp.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_e2e_log(path):
    """Return a dict with miou/aacc/peak_alloc/peak_reserved/ms_per_img/images."""
    if not osp.exists(path):
        return None
    info = {}
    first = last = None
    for line in open(path, errors="replace"):
        m = ITER_RE.search(line)
        if m:
            mo, d, h, mi, s, n, total = (int(g) for g in m.groups())
            stamp = datetime(2026, mo, d, h, mi, s)
            if first is None:
                first = (stamp, n)
            last = (stamp, n)
            info["images"] = total
        if line.startswith("RESULT "):
            for key in ("mIoU", "aAcc", "mAcc", "selectedK", "candidateK",
                        "roundsRun", "s2c2Seconds"):
                m2 = re.search(r"\b%s=([0-9.]+)" % key, line)
                if m2:
                    info[key] = float(m2.group(1))
        if line.startswith("PEAKMEM "):
            m3 = re.search(r"allocated_mb=(\d+) reserved_mb=(\d+)", line)
            if m3:
                info["peak_alloc_mb"] = int(m3.group(1))
                info["peak_reserved_mb"] = int(m3.group(2))
    if first and last and last[1] > first[1]:
        seconds = (last[0] - first[0]).total_seconds()
        info["ms_per_img"] = seconds * 1000.0 / (last[1] - first[1])
    return info or None


def fig_vocab(sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = sorted((r for r in sweep if r["mode"] == "twostage"),
                 key=lambda r: int(r["num_classes"]))
    noc = sorted((r for r in sweep if r["mode"] == "nocss"),
                 key=lambda r: int(r["num_classes"]))
    if not two or not noc:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in (ax1, ax2):
        ax.grid(True, color="#e5e7eb", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    x2 = [int(r["num_classes"]) for r in two]
    y2 = [float(r["total_r0_ms"]) for r in two]
    xn = [int(r["num_classes"]) for r in noc]
    yn = [float(r["total_r0_ms"]) for r in noc]
    ax1.plot(xn, yn, "-o", color=ORANGE, linewidth=2, markersize=5,
             label="Full-vocab (no CSS)")
    ax1.plot(x2, y2, "-o", color=BLUE, linewidth=2, markersize=5,
             label="S2C2Seg++ two-stage (CSS)")
    ax1.annotate("no CSS", (xn[-1], yn[-1]), textcoords="offset points",
                 xytext=(-8, 8), color=ORANGE, ha="right")
    ax1.annotate("with CSS", (x2[-1], y2[-1]), textcoords="offset points",
                 xytext=(-8, -14), color=BLUE, ha="right")
    ax1.set_xlabel("Vocabulary size |C|")
    ax1.set_ylabel("Latency per 336x336 window (ms)")
    ax1.set_title("(a) Vocabulary-scaling cost (reference two-stage)")
    ax1.legend(frameon=False, loc="upper left")

    cp = [float(r["cprime_median"]) for r in two]
    ax2.plot(x2, x2, "--", color=GRAY, linewidth=1.2, label="|C'| = |C|")
    ax2.plot(x2, cp, "-o", color=GREEN, linewidth=2, markersize=5,
             label="Selected |C'| (median)")
    ax2.set_xlabel("Vocabulary size |C|")
    ax2.set_ylabel("Selected subset size |C'|")
    ax2.set_title("(b) CSS compression ratio")
    ax2.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(osp.join(OUT, "fig_vocab_scaling.png"), dpi=160)
    plt.close(fig)
    return True


def main():
    lines = ["# Efficiency benchmark report", "",
             "Hardware: a single NVIDIA RTX 4090. "
             "Measurements are stored in the CSV files alongside this report.", "",
             "Two measurement schedules are reported. The **two-stage "
             "reference schedule** (`scripts/bench_efficiency.py`) runs fp16 "
             "per-prompt sequential CLIPSeg and re-encodes the selected "
             "subset after CSS. The **released default schedule** (`eval.py`) "
             "evaluates CLIPSeg once over the full vocabulary, caches the "
             "predictions, and lets refinement rounds re-rank/re-fuse the "
             "cache under the standard slide-inference protocol; all mIoU "
             "values come from this schedule.", ""]

    # ---- Experiment 1: vocabulary-scaling sweep ----
    sweep = read_csv(osp.join(OUT, "vocab_scaling.csv"))
    lines += ["## Experiment 1: vocabulary-scaling cost "
              "(two-stage reference schedule, median over 20 images)", ""]
    if sweep:
        ok = fig_vocab(sweep)
        if ok:
            lines += ["![vocab scaling](fig_vocab_scaling.png)", ""]
        lines += ["| Vocabulary | \\|C\\| | \\|C'\\| | Total w/o CSS (ms) "
                  "| Total w/ CSS (ms) | CSS rel. overhead "
                  "| CSS selection (ms) | Peak mem (MB) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        by_tag = {}
        for r in sweep:
            by_tag.setdefault(r["tag"], {})[r["mode"]] = r
        for tag, modes in sorted(
                by_tag.items(),
                key=lambda kv: int(next(iter(kv[1].values()))["num_classes"])):
            t, n = modes.get("twostage"), modes.get("nocss")
            if not (t and n):
                continue
            tt, tn = float(t["total_r0_ms"]), float(n["total_r0_ms"])
            lines.append(
                "| %s | %s | %s | %.0f | %.0f | %+.0f%% | %s | %s |"
                % (tag, t["num_classes"], t["cprime_median"], tn, tt,
                   (tt - tn) / tn * 100, t["CSS selection"],
                   t["peak_alloc_mb"]))
        lines.append("")
    else:
        lines += ["(vocab_scaling.csv missing; section skipped)", ""]

    # ---- Experiment 2: FLOPs ----
    lines += ["## Experiment 2: FLOPs accounting", ""]
    units = read_csv(osp.join(OUT, "flops_units.csv"))
    if units:
        from scripts.bench_flops import format_report
        report = format_report({r["unit"]: float(r["gflops"]) for r in units})
        with open(osp.join(OUT, "flops.md"), "w") as f:
            f.write(report + "\n")
        body = report.split("\n", 2)[2]
        lines += [("\n" + body).replace("\n## ", "\n### ").lstrip("\n"), ""]
    else:
        lines += ["(flops_units.csv missing; section skipped)", ""]

    # ---- Experiment 3: end-to-end runs ----
    e2e = {}
    for row in read_csv(osp.join(OUT, "e2e_runs.csv")):
        if row["name"] not in E2E_META or row["rc"] != "0":
            continue
        parsed = {}
        for key in ("ms_per_img", "mIoU", "wall_s"):
            if row.get(key):
                parsed[key] = float(row[key])
        for key in ("images", "peak_alloc_mb"):
            if row.get(key):
                parsed[key] = int(row[key])
        if parsed.get("ms_per_img") and "mIoU" in parsed:
            e2e[row["name"]] = parsed

    lines += ["## Experiment 3: end-to-end latency / throughput under the "
              "full evaluation protocol (same harness; evaluation-loop "
              "timing, model loading excluded)", ""]
    if e2e:
        lines += ["| Protocol | Config | Images | ms/img | img/s | vs. base "
                  "| mIoU | Peak mem (MB) | Wall clock (min) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        base_ms = {}
        for name, (proto, variant, is_base) in E2E_META.items():
            if is_base and name in e2e and e2e[name].get("ms_per_img"):
                base_ms[proto] = e2e[name]["ms_per_img"]
        for name, (proto, variant, is_base) in E2E_META.items():
            d = e2e.get(name)
            if not d:
                continue
            ms = d.get("ms_per_img")
            rel = ("%.1f×" % (ms / base_ms[proto])
                   if ms and proto in base_ms else "—")
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    proto, variant, d.get("images", "—"),
                    "%.1f" % ms if ms else "—",
                    "%.2f" % (1000.0 / ms) if ms else "—", rel,
                    "%.2f" % d["mIoU"] if "mIoU" in d else "—",
                    d.get("peak_alloc_mb", "—"),
                    "%.0f" % (d["wall_s"] / 60) if "wall_s" in d else "—"))
        lines.append("")
    else:
        lines += ["(complete end-to-end CSV rows missing; section skipped)", ""]

    lines += [
        "## Measurement notes", "",
        "- FLOPs totals use the saved unit costs, rounded to 0.001 GFLOPs.",
        "- End-to-end rows contain measured evaluation-loop latency and mIoU; "
        "the report can be rebuilt from the CSV files without evaluation logs. "
        "A dash in the wall-clock column means total command duration was not recorded.",
        "- Experiment 1 uses the two-stage reference schedule to isolate "
        "the vocabulary-compression effect of CSS; Experiment 3 uses the "
        "released default schedule, so mIoU and latency come from the same "
        "runs and are directly comparable.",
        "- In the released default schedule CLIPSeg runs once over the full "
        "vocabulary (one vision pass per window plus a per-prompt decoder "
        "pass) and refinement rounds re-fuse the cached predictions, which "
        "is why the per-round marginal cost differs between the two "
        "schedules."]
    text = "\n".join(lines)
    with open(osp.join(OUT, "EFFICIENCY_REPORT.md"), "w") as f:
        f.write(text + "\n")
    print(text)


E2E_FIELDS = ["name", "config", "wall_s", "rc", "images", "ms_per_img",
              "mIoU", "peak_alloc_mb"]


def record_run(values, log_path):
    name, config, wall_s, rc = values
    if name not in E2E_META:
        raise ValueError("unknown end-to-end run: %s" % name)
    row = dict(name=name, config=config, wall_s=wall_s, rc=rc)
    parsed = parse_e2e_log(log_path) or {}
    row.update({key: parsed[key] for key in E2E_FIELDS if key in parsed})
    path = osp.join(OUT, "e2e_runs.csv")
    rows = [old for old in read_csv(path) if old["name"] != name]
    rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=E2E_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", nargs=4, metavar=("NAME", "CONFIG", "WALL_S", "RC"))
    parser.add_argument("--log", help="evaluation log to parse when recording a run")
    args = parser.parse_args()
    if args.record:
        if not args.log:
            parser.error("--record requires --log")
        record_run(args.record, args.log)
    else:
        main()
