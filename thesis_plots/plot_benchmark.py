#!/usr/bin/env python3
"""Thesis figures for the full CPU/stack benchmark — detailed, per tier.

Stats use min / mean / max (n=20 per cell, so percentiles aren't meaningful).

Per tier → figures/full_benchmark/<tier>/:
  cpu_gpu_donut   overall CPU vs GPU share of request time (the headline)
  cpu_vs_gpu      CPU (frontend) vs GPU (vLLM) per cell — log scale, both visible
  pipeline        CPU pipeline only (embed/Milvus/SeaweedFS/Mongo) — ms scale, small times visible
  decomposition   full stacked request time incl. vLLM
  latency         e2e per cell — mean bar, min–max whiskers
  streaming       TTFT / TPOT per cell (mean + min–max)  (skipped if the tier wasn't streamed)
  tma             top-down slot breakdown per pod  — the cross-pod microarchitecture takeaway
  ipc             instructions-per-cycle per pod

Cross-tier / cross-pod → figures/full_benchmark/cross_tier/:
  vllm_pod_cpu    vLLM pod CPU-utilised % and context-switches across tiers (it is GPU-bound)
  tpot_compare    TPOT per cell across tiers (tok64 inflated → tok320 true)
  ttft_compare    TTFT per cell across tiers

Usage:  python3 thesis_plots/plot_benchmark.py [RUN_DIR] [--path rag]
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTROOT = Path(__file__).resolve().parent / "figures" / "full_benchmark"
TIERS = ["tok64", "tok192", "tok320"]
CELL_ORDER = ["rag", "rag_pure_fetch", "llm_direct", "sc_a", "sc_b"]
SIZES = ["short", "medium", "long", "very_long"]


# ── parsing / stats ──────────────────────────────────────────────────────────
def _vals(rows, col):
    return [float(r[col]) for r in rows if r.get(col) not in ("", None)]


def _stats(rows, col):
    v = _vals(rows, col)
    if not v:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {"min": min(v), "mean": st.mean(v), "max": max(v)}


def _med(rows, col):
    v = _vals(rows, col)
    return st.median(v) if v else 0.0


def load_cells(run, tier):
    cells = {}
    for d in sorted(glob.glob(str(run / tier / "cell_*"))):
        name = Path(d).name[len("cell_"):]
        # measurement csv is "<cell>_<timestamp>.csv"; exclude the vllm_metrics
        # time-series csv (added by the scraper) and the per-pass subdir csvs.
        csvs = sorted(c for c in glob.glob(f"{d}/*.csv")
                      if "/pass" not in c and "vllm_metrics" not in Path(c).name)
        if not csvs:
            continue
        with open(csvs[0]) as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("http_status") in ("200", "", None)]
        if rows:
            cells[name] = rows
    return cells


def order_cells(names):
    def key(n):
        for i, base in enumerate(CELL_ORDER):
            if n.startswith(base) and (n == base or n[len(base):].lstrip("_") in SIZES):
                for j, s in enumerate(SIZES):
                    if n.endswith(s):
                        return (i, j)
                return (i, 0)
        return (99, 0)
    return sorted(names, key=key)


def _counter(path, name):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        m = re.match(rf"\s*([0-9]+)\s+{re.escape(name)}(\s|$)", line)
        if m:
            return float(m.group(1))
    return None


def _vllm_cpu(run, tier, cell):
    f = run / tier / f"cell_{cell}" / "perf_pass1_vllm.txt"
    if not f.exists():
        return None
    s = f.read_text()
    cpu = re.search(r"([0-9.]+)\s+CPUs utilized", s)
    ctx = re.search(r"([0-9]+)\s+context-switches", s)
    return {"cpu_util": float(cpu.group(1)) * 100 if cpu else 0.0,
            "ctx": float(ctx.group(1)) if ctx else 0.0}


def _components(rows):
    embed = _med(rows, "rag_embed_ms") + _med(rows, "cache_embed_ms")
    milvus = _med(rows, "rag_milvus_ms") + _med(rows, "cache_milvus_ms")
    seaweed = _med(rows, "rag_seaweed_ms")
    mongo = _med(rows, "cache_mongo_ms")
    vllm = _med(rows, "model_backend_http_ms")
    e2e = _med(rows, "e2e_ms")
    other = max(0.0, e2e - (embed + milvus + seaweed + mongo + vllm))
    return dict(embed=embed, milvus=milvus, seaweed=seaweed, mongo=mongo, vllm=vllm, other=other, e2e=e2e)


PIPE = [("embed", "BGE embed", style.C["green"]), ("milvus", "Milvus", style.C["orange"]),
        ("seaweed", "SeaweedFS", style.C["yellow"]), ("mongo", "MongoDB", style.C["purple"])]


# ── per-tier figures ─────────────────────────────────────────────────────────
def fig_cpu_gpu_donut(cells, tier, out):
    cpu = sum(sum(_vals(rows, "frontend_overhead_ms")) for rows in cells.values())
    gpu = sum(sum(_vals(rows, "model_backend_http_ms")) for rows in cells.values())
    tot = cpu + gpu or 1
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    wedges, _ = ax.pie([gpu, cpu], colors=[style.C["red"], style.C["blue"]],
                       startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0.08, "GPU", ha="center", fontsize=12, fontweight="bold", color=style.C["red"])
    ax.text(0, -0.12, f"{100*gpu/tot:.1f}%", ha="center", fontsize=14, fontweight="bold")
    ax.legend(wedges, [f"GPU (vLLM)  {100*gpu/tot:.1f}%", f"CPU (frontend)  {100*cpu/tot:.2f}%"],
              loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9)
    ax.set_title(f"Where request time goes ({tier})")
    style.save(fig, str(out / "cpu_gpu_donut"))


def fig_cpu_vs_gpu(cells, tier, out):
    names = order_cells(cells)
    cpu = [_med(cells[n], "frontend_overhead_ms") for n in names]
    gpu = [_med(cells[n], "model_backend_http_ms") for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(x - 0.2, np.maximum(cpu, 1e-3), 0.4, label="CPU (frontend)", color=style.C["blue"])
    ax.bar(x + 0.2, np.maximum(gpu, 1e-3), 0.4, label="GPU (vLLM)", color=style.C["red"])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Time per request (ms, log)")
    ax.set_title(f"CPU vs GPU time per cell ({tier})")
    ax.legend()
    style.save(fig, str(out / "cpu_vs_gpu"))


def fig_pipeline(cells, tier, out):
    names = [n for n in order_cells(cells) if any(_components(cells[n])[k] > 0 for k, _, _ in PIPE)]
    if not names:
        return
    comp = {n: _components(cells[n]) for n in names}
    x = np.arange(len(names)); fig, ax = plt.subplots(figsize=(7.2, 3.5)); bottom = np.zeros(len(names))
    for key, lab, col in PIPE:
        vals = np.array([comp[n][key] for n in names])
        ax.bar(x, vals, 0.7, bottom=bottom, label=lab, color=col); bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CPU pipeline time (ms)")
    ax.set_title(f"Frontend pipeline cost per cell ({tier}) — GPU excluded")
    ax.legend(ncol=4, fontsize=8)
    style.save(fig, str(out / "pipeline"))


def fig_decomposition(cells, tier, out):
    names = order_cells(cells); comp = {n: _components(cells[n]) for n in names}
    labels = [("embed", "BGE embed", style.C["green"]), ("milvus", "Milvus", style.C["orange"]),
              ("seaweed", "SeaweedFS", style.C["yellow"]), ("mongo", "MongoDB", style.C["purple"]),
              ("vllm", "vLLM (GPU)", style.C["red"]), ("other", "other CPU", style.C["grey"])]
    x = np.arange(len(names)); fig, ax = plt.subplots(figsize=(7.4, 3.7)); bottom = np.zeros(len(names))
    for key, lab, col in labels:
        vals = np.array([comp[n][key] / 1000 for n in names])
        ax.bar(x, vals, 0.7, bottom=bottom, label=lab, color=col); bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Time (s)"); ax.set_title(f"Request time decomposition ({tier})")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    style.save(fig, str(out / "decomposition"))


def _meanbar(ax, x, off, w, stat_list, color, label, scale=1.0):
    mean = np.array([s["mean"] * scale for s in stat_list])
    lo = mean - np.array([s["min"] * scale for s in stat_list])
    hi = np.array([s["max"] * scale for s in stat_list]) - mean
    ax.bar(x + off, mean, w, color=color, label=label, yerr=[lo, hi],
           capsize=2, error_kw=dict(lw=0.8, alpha=0.6))


def fig_latency(cells, tier, out):
    names = order_cells(cells)
    stats = [_stats(cells[n], "e2e_ms") for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    _meanbar(ax, x, 0, 0.6, stats, style.C["blue"], "mean (whiskers = min–max)", scale=1 / 1000)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("End-to-end latency (s)")
    ax.set_title(f"Request latency per cell ({tier}) — mean, min–max")
    ax.legend()
    style.save(fig, str(out / "latency"))


def fig_streaming(cells, tier, out):
    names = [n for n in order_cells(cells) if _med(cells[n], "ttft_ms") > 0]
    if not names:
        print(f"    ({tier}: not streamed — skipping streaming)")
        return
    ttft = [_stats(cells[n], "ttft_ms") for n in names]
    tpot = [_stats(cells[n], "tpot_ms") for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    _meanbar(ax, x, -0.2, 0.4, ttft, style.C["green"], "TTFT (ms)")
    _meanbar(ax, x, 0.2, 0.4, tpot, style.C["orange"], "TPOT (ms/token)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ms"); ax.set_title(f"Streaming latency ({tier}) — mean, min–max")
    ax.legend()
    style.save(fig, str(out / "streaming"))


def fig_prefill_vs_decode(cells, tier, out):
    """Share of GPU time spent in prefill (TTFT) vs decode (generation), per cell."""
    names = [n for n in order_cells(cells)
             if _med(cells[n], "generation_ms") > 0 and _med(cells[n], "ttft_ms") > 0]
    if not names:
        print(f"    ({tier}: not streamed — skipping prefill_vs_decode)")
        return
    prefill = np.array([_med(cells[n], "ttft_ms") for n in names])
    decode = np.array([_med(cells[n], "generation_ms") for n in names])
    tot = prefill + decode
    pf, dc = 100 * prefill / tot, 100 * decode / tot
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(x, dc, 0.7, label="Decode (generation)", color=style.C["red"])
    ax.bar(x, pf, 0.7, bottom=dc, label="Prefill (TTFT)", color=style.C["green"])
    for i, p in enumerate(pf):
        ax.text(x[i], 101, f"{p:.1f}%", ha="center", fontsize=7, color=style.C["green"])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("% of GPU time"); ax.set_ylim(0, 110)
    ax.set_title(f"Prefill vs decode share of GPU time ({tier}) — decode dominates")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.42))
    style.save(fig, str(out / "prefill_vs_decode"))


def _tma_file(run, kind, path, pod, tier):
    """run_benchmark names TMA files with NO tier suffix for tok64
    (tma_slots_rag_fastapi.txt) and a '_<tier>' suffix for tok192/tok320.
    Return whichever exists (suffixed preferred)."""
    base = run / "tma"
    suf = base / f"tma_{kind}_{path}_{pod}_{tier}.txt"
    if suf.exists():
        return suf
    return base / f"tma_{kind}_{path}_{pod}.txt"


def fig_tma(run, tier, path, out):
    pods = ["fastapi", "milvus", "mongodb", "seaweed_filer", "seaweed_volume", "llmd_gateway", "vllm"]
    keys = [("topdown-retiring", "Retiring", style.C["green"]),
            ("topdown-fe-bound", "Frontend-bound", style.C["sky"]),
            ("topdown-bad-spec", "Bad speculation", style.C["orange"]),
            ("topdown-be-bound", "Backend-bound", style.C["red"])]
    data, labels = [], []
    for pod in pods:
        f = _tma_file(run, "slots", path, pod, tier)
        slots = _counter(f, "slots")
        if slots:
            labels.append(pod); data.append([(_counter(f, k) or 0) / slots * 100 for k, _, _ in keys])
    if not data:
        return
    data = np.array(data); x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(7.0, 3.7)); bottom = np.zeros(len(labels))
    for i, (_, lab, col) in enumerate(keys):
        ax.bar(x, data[:, i], 0.6, bottom=bottom, label=lab, color=col); bottom += data[:, i]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0, 100)
    ax.set_title(f"Where each pod's CPU stalls — TMA top-down ({path}, {tier})")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    style.save(fig, str(out / "tma"))


def fig_ipc(run, tier, out):
    cell = "rag_medium"; pods = ["fastapi", "milvus", "mongodb", "seaweed_filer", "seaweed_volume", "llmd_gateway"]
    labels, ipc = [], []
    for pod in pods:
        f = run / tier / f"cell_{cell}" / f"perf_pass1_{pod}.txt"
        c, ins = _counter(f, "cycles"), _counter(f, "instructions")
        if c and ins:
            labels.append(pod); ipc.append(ins / c)
    if not ipc:
        return
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, ipc, 0.6, color=style.C["blue"])
    for i, v in enumerate(ipc):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IPC (instructions / cycle)"); ax.set_title(f"Per-pod IPC ({cell}, {tier})")
    style.save(fig, str(out / "ipc"))


# Regime keyed by the dominant top-down bucket (from the canonical PERF_METRICS
# slots counters — more reliable than toplev's multiplexed leaf). Each entry:
# (regime label, colour, short bottleneck tag).
REGIME = {
    "exec": ("Execution-bound\n(compute)", style.C["green"], "exec: ports"),
    "mem":  ("Memory-latency-bound", style.C["orange"], "mem: L3"),
    "fe":   ("Frontend-bound\n(instruction)", style.C["sky"], "frontend: fetch"),
}


def _toplev_node(path, name):
    """Value (% column) of a named TMA node in a toplev file, or None."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if re.search(rf"{re.escape(name)}\s+%", line):
            m = re.search(r"%\s+\S+\s+([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def _core_frac(path):
    """Fraction of backend-bound that is Core (execution) vs Memory.
    Core/Memory lines are pruned by toplev when below threshold → derive from
    Backend − the present sibling."""
    be = _toplev_node(path, "Backend_Bound")
    c = _toplev_node(path, "Backend_Bound.Core_Bound")
    m = _toplev_node(path, "Backend_Bound.Memory_Bound")
    if c is None and m is not None and be is not None:
        c = max(be - m, 0.0)
    if m is None and c is not None and be is not None:
        m = max(be - c, 0.0)
    if not c and not m:
        return 0.5
    return c / (c + m)


def fig_micro_summary(run, tier, path, out):
    """One-slide CPU story: each pod's TMA breakdown with the backend split into
    execution- vs memory-bound, grouped into the three microarchitecture
    regimes, annotated with its bottleneck node + IPC."""
    pods = ["fastapi", "milvus", "seaweed_filer", "seaweed_volume", "mongodb", "llmd_gateway"]
    pretty = {"fastapi": "fastapi (BGE embed)", "milvus": "milvus (vector search)",
              "seaweed_filer": "seaweed_filer", "seaweed_volume": "seaweed_volume",
              "mongodb": "mongodb", "llmd_gateway": "llmd_gateway (router)"}
    # segments left→right; backend split into memory then execution (the regime cue)
    segs = [("Retiring", style.C["green"]), ("Frontend-bound", style.C["sky"]),
            ("Bad spec.", style.C["grey"]), ("Backend · memory", style.C["orange"]),
            ("Backend · execution", style.C["red"])]
    rows = []
    for pod in pods:
        f = _tma_file(run, "slots", path, pod, tier)
        slots = _counter(f, "slots")
        if not slots:
            continue
        tl = _tma_file(run, "toplev", path, pod, tier)
        ret = (_counter(f, "topdown-retiring") or 0) / slots * 100
        fe = (_counter(f, "topdown-fe-bound") or 0) / slots * 100
        bad = (_counter(f, "topdown-bad-spec") or 0) / slots * 100
        be = (_counter(f, "topdown-be-bound") or 0) / slots * 100
        cf = _core_frac(tl)               # core/(core+mem) split of backend
        be_mem, be_core = be * (1 - cf), be * cf
        # classify by the largest "bound" bucket — the canonical counters decide,
        # not toplev's (multiplexed, less reliable) <== leaf.
        buckets = {"exec": be_core, "mem": be_mem, "fe": fe}
        key = max(buckets, key=buckets.get)
        regime, rcol, short = REGIME[key]
        # mechanism counter backing the regime, from the cell's perf passes
        cell = run / tier / "cell_rag_medium"

        def pc(name):
            for pp in ("perf_pass1", "perf_pass2a", "perf_pass2b", "perf_pass4"):
                v = _counter(cell / f"{pp}_{pod}.txt", name)
                if v is not None:
                    return v
            return None
        c, ins = pc("cycles"), pc("instructions")
        ki = ins / 1000 if ins else None
        # pass2a no longer collects cache-misses; derive LLC MPKI from the
        # load-outcome hierarchy (mem_load_retired.l3_miss = load LLC misses).
        l1i = pc("L1-icache-load-misses")
        llc = pc("mem_load_retired.l3_miss")
        uops = pc("uops_executed.core")
        if key == "fe":
            ev = f"L1-I$ miss: {l1i / ki:.0f} MPKI" if l1i and ki else ""
        elif key == "mem":
            ev = f"LLC load miss: {llc / ki:.1f} MPKI" if llc and ki else ""
        else:
            ev = f"exec: {uops / c:.2f} µops/cycle" if uops and c else ""
        rows.append({"pod": pod, "frac": [ret, fe, bad, be_mem, be_core],
                     "bn": short, "measure": buckets[key], "ev": ev,
                     "ipc": ins / c if c and ins else None,
                     "regime": regime, "rcol": rcol})
    if not rows:
        return
    n = len(rows)
    y = np.arange(n)[::-1]          # first pod at the top
    fr = np.array([r["frac"] for r in rows])
    fig, ax = plt.subplots(figsize=(11.8, 0.95 * n + 2.0))

    # regime background bands + label (groups of consecutive same-regime rows)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and rows[j + 1]["regime"] == rows[i]["regime"]:
            j += 1
        ax.axhspan(y[j] - 0.5, y[i] + 0.5, color=rows[i]["rcol"], alpha=0.10, zorder=0)
        ax.text(218, (y[i] + y[j]) / 2, rows[i]["regime"], va="center", ha="center",
                fontsize=9.5, fontweight="bold", color=rows[i]["rcol"], clip_on=False)
        i = j + 1

    left = np.zeros(n)
    for k, (lab, col) in enumerate(segs):
        ax.barh(y, fr[:, k], 0.52, left=left, label=lab, color=col, zorder=3)
        left += fr[:, k]

    # column headers + per-row bottleneck (% of slots) + mechanism counter + IPC
    ax.text(108, y[0] + 0.62, "bottleneck  /  mechanism counter", fontsize=8.5,
            style="italic", color=style.C["grey"], clip_on=False)
    ax.text(182, y[0] + 0.62, "IPC", fontsize=8.5, style="italic",
            color=style.C["grey"], clip_on=False)
    for idx, r in enumerate(rows):
        ipc = f"{r['ipc']:.2f}" if r["ipc"] is not None else "–"
        meas = f"  —  {r['measure']:.0f}% of slots" if r["measure"] is not None else ""
        ax.text(108, y[idx] + 0.16, f"⟶ {r['bn']}{meas}", va="center", ha="left",
                fontsize=9, fontweight="bold", color=r["rcol"], clip_on=False)
        if r["ev"]:
            ax.text(120, y[idx] - 0.20, r["ev"], va="center", ha="left",
                    fontsize=8.5, color=r["rcol"], clip_on=False)
        ax.text(182, y[idx], ipc, va="center", ha="left",
                fontsize=12, fontweight="bold", color="black", clip_on=False)

    fig.text(0.5, -0.01,
             "Pod classified by its largest top-down bucket (% of CPU pipeline slots). "
             "Below each: the hardware counter for that mechanism — MPKI = cache misses per "
             "1000 instructions (L1-I$ = instruction cache, LLC = last-level cache), or execution µops/cycle.",
             ha="center", fontsize=7.5, style="italic", color=style.C["grey"])

    ax.set_yticks(y); ax.set_yticklabels([pretty[r["pod"]] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of pipeline slots (TMA top-down)")
    ax.set_title(f"CPU microarchitecture regimes across the service pods ({path}, {tier})")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.42, -0.16),
              columnspacing=1.0, handletextpad=0.4)
    style.save(fig, str(out / "micro_summary"))


# ── cross-tier / cross-pod ───────────────────────────────────────────────────
def fig_vllm_pod_cpu(run, out):
    cell = "llm_direct_medium"
    cpu = [(_vllm_cpu(run, t, cell) or {}).get("cpu_util", 0) for t in TIERS]
    ctx = [(_vllm_cpu(run, t, cell) or {}).get("ctx", 0) for t in TIERS]
    x = np.arange(len(TIERS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    a1.bar(x, cpu, 0.55, color=style.C["blue"])
    for i, v in enumerate(cpu):
        a1.text(i, v + 0.05, f"{v:.1f}%", ha="center", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(TIERS); a1.set_ylabel("CPU utilised (%)")
    a1.set_ylim(0, max(cpu) * 1.6 + 0.5); a1.set_title("vLLM pod is CPU-idle (≈1%)")
    a2.bar(x, ctx, 0.55, color=style.C["red"])
    for i, v in enumerate(ctx):
        a2.text(i, v + max(ctx) * 0.02, f"{int(v)}", ha="center", fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(TIERS); a2.set_ylabel("context-switches")
    a2.set_title("…but switches scale with output (waiting on GPU)")
    fig.suptitle("vLLM is GPU-bound: the CPU just blocks per token", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    style.save(fig, str(out / "vllm_pod_cpu"))


def fig_metric_compare(run, metric, ylabel, title, fname, out):
    cs = {t: load_cells(run, t) for t in TIERS}
    ref = next((t for t in ("tok320", "tok192", "tok64") if cs.get(t)), None)
    if not ref:
        return
    names = order_cells(cs[ref])
    fig, ax = plt.subplots(figsize=(8.0, 3.6)); x = np.arange(len(names)); w = 0.25; any_d = False
    for j, t in enumerate(TIERS):
        vals = [_med(cs[t][n], metric) if n in cs[t] else 0 for n in names]
        if sum(vals) > 0:
            any_d = True
        ax.bar(x + (j - 1) * w, vals, w, label=t, color=style.CYCLE[j])
    if not any_d:
        plt.close(fig); return
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    style.save(fig, str(out / fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=str(ROOT / "benchmark_results" / "run_20260609_140052"))
    ap.add_argument("--path", default="rag")
    args = ap.parse_args()
    run = Path(args.run); style.set_style()
    print(f"benchmark run: {run.name}")
    for tier in TIERS:
        out = OUTROOT / tier; out.mkdir(parents=True, exist_ok=True)
        cells = load_cells(run, tier)
        if not cells:
            continue
        print(f"  {tier}: {len(cells)} cells")
        fig_cpu_gpu_donut(cells, tier, out)
        fig_cpu_vs_gpu(cells, tier, out)
        fig_pipeline(cells, tier, out)
        fig_decomposition(cells, tier, out)
        fig_latency(cells, tier, out)
        fig_streaming(cells, tier, out)
        fig_prefill_vs_decode(cells, tier, out)
        fig_tma(run, tier, args.path, out)
        fig_ipc(run, tier, out)
    cx = OUTROOT / "cross_tier"; cx.mkdir(parents=True, exist_ok=True); print("  cross_tier")
    # micro_summary wants the most stable TMA windows (longest drives), so prefer
    # the highest tier available — tok64 drives are short (~9s) and the fastapi
    # uvicorn-worker PID can exit mid perf-attach, making tok64 TMA racy.
    msum_tier = next((t for t in ("tok320", "tok192", "tok64") if load_cells(run, t)), "tok64")
    fig_micro_summary(run, msum_tier, args.path, cx)
    fig_vllm_pod_cpu(run, cx)
    fig_metric_compare(run, "tpot_ms", "TPOT (ms/token)", "TPOT per cell across tiers", "tpot_compare", cx)
    fig_metric_compare(run, "ttft_ms", "TTFT (ms)", "TTFT per cell across tiers", "ttft_compare", cx)
    print(f"figures → {OUTROOT}")


if __name__ == "__main__":
    main()
