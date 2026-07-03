#!/usr/bin/env python3
"""Thesis figures for a GPU prefill/decode sweep (GPU_benchmark run).

Figures (→ benchmark_results/plots/figures/gpu_sweep/):
  prefill_ttft   TTFT vs input length         — the compute-bound regime
  decode_tpot    TPOT vs output length        — the memory-bound regime (+ KV drift)
  regime         compute vs memory engine use — prefill vs decode, the headline contrast

Usage:
  python3 benchmark_results/plots/plot_gpu_sweep.py [RUN_DIR]
  (defaults to the latest GPU_benchmark/results/run_*)
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures" / "gpu_sweep"


def latest_run() -> Path:
    runs = sorted(glob.glob(str(ROOT / "GPU_benchmark" / "results" / "run_*")))
    if not runs:
        sys.exit("no GPU_benchmark/results/run_* found")
    return Path(runs[-1])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_points(run: Path, sweep: str) -> list[dict]:
    """Per-point latency aggregates (excluding the discarded first request)."""
    rows: dict[str, list[dict]] = {}
    with open(run / "all_requests.csv") as fh:
        for r in csv.DictReader(fh):
            if r["sweep"] != sweep or r["discarded"] == "1":
                continue
            rows.setdefault(r["point"], []).append(r)
    pts = []
    for point, rs in rows.items():
        if point.startswith("_"):
            continue
        in_tok = int(st.median(int(r["prompt_tokens"]) for r in rs))
        out_tok = int(st.median(int(r["completion_tokens"]) for r in rs))
        ttft = [_f(r["ttft_ms"]) for r in rs if _f(r["ttft_ms"]) is not None]
        tpot = [_f(r["tpot_ms"]) for r in rs if _f(r["tpot_ms"]) is not None]
        pts.append({
            "point": point, "in_tok": in_tok, "out_tok": out_tok,
            "ttft_med": st.median(ttft) if ttft else None,
            "tpot_med": st.median(tpot) if tpot else None,
            "dcgm": _dcgm(run, sweep, point),
        })
    return pts


def _dcgm(run: Path, sweep: str, point: str) -> dict:
    """Median/max across GPUs for the PROF engine + DRAM fields."""
    f = run / sweep / point / "dcgm_summary.json"
    out = {"engine_med": None, "engine_max": None, "dram_med": None, "dram_max": None}
    if not f.exists():
        return out
    series = json.loads(f.read_text()).get("series", {})

    def agg(needle, stat):
        vals = [v[stat] for k, v in series.items() if needle in k]
        return st.median(vals) if vals else None

    out["engine_med"] = agg("GR_ENGINE_ACTIVE", "median")
    out["engine_max"] = agg("GR_ENGINE_ACTIVE", "max")
    out["dram_med"] = agg("DRAM_ACTIVE", "median")
    out["dram_max"] = agg("DRAM_ACTIVE", "max")
    out["tensor_max"] = agg("PIPE_TENSOR_ACTIVE", "max")
    return out


def _vllm_kv(run: Path, sweep: str, point: str):
    """Max KV-cache utilisation (%) during a point, from vllm_summary.json."""
    f = run / sweep / point / "vllm_summary.json"
    if not f.exists():
        return None
    series = json.loads(f.read_text()).get("series", {})
    vals = [v["max"] for k, v in series.items() if "cache_usage_perc" in k]
    return max(vals) * 100 if vals else None


# ── Figures ──────────────────────────────────────────────────────────────────
def fig_prefill_ttft(pre: list[dict]) -> None:
    pre = sorted(pre, key=lambda p: p["in_tok"])
    x = [p["in_tok"] for p in pre]
    y = [p["ttft_med"] for p in pre]
    fig, ax = plt.subplots()
    ax.plot(x, y, "-o", color=style.C["blue"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Input length (prompt tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Prefill: time-to-first-token vs input length")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    style.save(fig, str(OUT / "prefill_ttft"))


def fig_decode_tpot(dec: list[dict]) -> None:
    dec = sorted(dec, key=lambda p: p["out_tok"])
    x = [p["out_tok"] for p in dec]
    y = [p["tpot_med"] for p in dec]
    fig, ax = plt.subplots()
    ax.plot(x, y, "-o", color=style.C["red"], label="measured TPOT")
    ax.axhline(y[0], ls="--", lw=1, color=style.C["grey"], label=f"baseline {y[0]:.1f} ms")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Output length (generated tokens)")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("Decode: per-token latency vs output length")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    # zoom y to show the small KV-cache drift
    ys = [v for v in y if v is not None]
    if ys:
        ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)
    ax.legend(loc="upper left")
    style.save(fig, str(OUT / "decode_tpot"))


def fig_regime(pre: list[dict], dec: list[dict]) -> None:
    """Headline: compute-engine vs HBM utilisation, prefill vs decode."""
    def avg(pts, key, stat):
        # prefill: use long points (median reliable); decode: all points.
        vals = [p["dcgm"][f"{key}_{stat}"] for p in pts if p["dcgm"][f"{key}_{stat}"] is not None]
        return st.mean(vals) * 100 if vals else 0

    pre_long = [p for p in pre if p["in_tok"] >= 1500]
    groups = ["Prefill\n(compute-bound)", "Decode\n(memory-bound)"]
    engine = [avg(pre_long, "engine", "max"), avg(dec, "engine", "med")]
    dram = [avg(pre_long, "dram", "med"), avg(dec, "dram", "med")]

    fig, ax = plt.subplots()
    import numpy as np
    xpos = np.arange(len(groups))
    w = 0.36
    ax.bar(xpos - w / 2, engine, w, label="Compute engine active", color=style.C["green"])
    ax.bar(xpos + w / 2, dram, w, label="HBM (DRAM) active", color=style.C["orange"])
    for i, (e, d) in enumerate(zip(engine, dram)):
        ax.text(i - w / 2, e + 1.5, f"{e:.0f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, d + 1.5, f"{d:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("GPU utilisation (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Prefill is compute-bound, decode is memory-bound")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    style.save(fig, str(OUT / "regime"))


def fig_prefill_dcgm(pre: list[dict]) -> None:
    pre = sorted(pre, key=lambda p: p["in_tok"])
    x = [p["in_tok"] for p in pre]
    eng = [(p["dcgm"]["engine_max"] or 0) * 100 for p in pre]
    dram = [(p["dcgm"]["dram_max"] or 0) * 100 for p in pre]
    fig, ax = plt.subplots()
    ax.plot(x, eng, "-o", color=style.C["green"], label="Compute engine")
    ax.plot(x, dram, "-s", color=style.C["orange"], label="HBM (DRAM)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Input length (tokens)"); ax.set_ylabel("GPU active (peak %)"); ax.set_ylim(0, 105)
    ax.set_title("Prefill GPU utilisation — compute-bound")
    ax.legend(loc="center right")
    style.save(fig, str(OUT / "prefill_dcgm"))


def fig_decode_dcgm(dec: list[dict]) -> None:
    dec = sorted(dec, key=lambda p: p["out_tok"])
    x = [p["out_tok"] for p in dec]
    eng = [(p["dcgm"]["engine_med"] or 0) * 100 for p in dec]
    dram = [(p["dcgm"]["dram_med"] or 0) * 100 for p in dec]
    fig, ax = plt.subplots()
    ax.plot(x, eng, "-o", color=style.C["green"], label="Compute engine")
    ax.plot(x, dram, "-s", color=style.C["orange"], label="HBM (DRAM)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Output length (tokens)"); ax.set_ylabel("GPU active (median %)"); ax.set_ylim(0, 105)
    ax.set_title("Decode GPU utilisation — memory-bound (HBM ~83%)")
    ax.legend(loc="center right")
    style.save(fig, str(OUT / "decode_dcgm"))


def fig_decode_kvcache(dec: list[dict], run: Path) -> None:
    dec = sorted(dec, key=lambda p: p["out_tok"])
    x = [p["out_tok"] for p in dec]
    kv = [_vllm_kv(run, "decode", p["point"]) for p in dec]
    if not any(k for k in kv):
        return
    fig, ax = plt.subplots()
    ax.plot(x, kv, "-o", color=style.C["purple"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(x); ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Output length (tokens)"); ax.set_ylabel("KV-cache utilisation (peak %)")
    ax.set_title("Decode: KV-cache grows with output (drives the TPOT rise)")
    style.save(fig, str(OUT / "decode_kvcache"))


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    OUT.mkdir(parents=True, exist_ok=True)
    style.set_style()
    print(f"GPU sweep run: {run.name}")
    pre = load_points(run, "prefill")
    dec = load_points(run, "decode")
    if pre:
        fig_prefill_ttft(pre)
        fig_prefill_dcgm(pre)
    if dec:
        fig_decode_tpot(dec)
        fig_decode_dcgm(dec)
        fig_decode_kvcache(dec, run)
    if pre and dec:
        fig_regime(pre, dec)
    print(f"figures → {OUT}")


if __name__ == "__main__":
    main()
