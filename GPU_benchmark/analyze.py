#!/usr/bin/env python3
"""Summarise a GPU sweep run into prefill and decode tables.

Pulls per-point latency (median TTFT / TPOT, excluding the discarded first
request) from all_requests.csv, and the GPU/engine signals from each point's
dcgm_summary.json / vllm_summary.json.

Usage:
  python3 analyze.py results/run_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _median(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def _load_rows(run: Path) -> List[dict]:
    rows = []
    with open(run / "all_requests.csv") as f:
        for r in csv.DictReader(f):
            if r.get("discarded") == "1":
                continue
            for k in ("ttft_ms", "tpot_ms", "generation_ms", "e2e_ms"):
                r[k] = float(r[k]) if r.get(k) not in ("", None) else None
            for k in ("prompt_tokens", "completion_tokens"):
                r[k] = int(r[k]) if r.get(k) not in ("", None) else 0
            rows.append(r)
    return rows


def _series_median(summary: Path, name_contains: str) -> Optional[float]:
    """Median across GPUs of the per-series median for a DCGM/vLLM metric."""
    if not summary.exists():
        return None
    data = json.loads(summary.read_text()).get("series", {})
    vals = [v["median"] for k, v in data.items() if name_contains in k]
    return round(statistics.median(vals), 4) if vals else None


def _point_dir(run: Path, sweep: str, label: str) -> Path:
    return run / sweep / label


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python3 analyze.py results/run_YYYYMMDD_HHMMSS")
    run = Path(sys.argv[1])
    info = json.loads((run / "run_info.json").read_text())
    rows = _load_rows(run)

    by_point: Dict[str, List[dict]] = {}
    for r in rows:
        by_point.setdefault((r["sweep"], r["point"]), []).append(r)

    print(f"\nGPU sweep — {info['model']}  ({info['timestamp']})")
    print(f"DCGM: {'on' if info.get('dcgm_enabled') else 'OFF (no GPU hw metrics)'}\n")

    # ── Prefill ──────────────────────────────────────────────────────────────
    pre = [(s, p) for (s, p) in by_point if s == "prefill" and not p.startswith("_")]
    if pre:
        print("PREFILL  (output=1, concurrency=1) — compute-bound regime")
        print(f"  {'in_tok':>7} {'TTFT(ms)':>9} {'tensor%':>8} {'sm%':>7} {'dram%':>7}")
        for s, p in sorted(pre, key=lambda x: int(x[1][2:])):
            grp = by_point[(s, p)]
            ttft = _median([r["ttft_ms"] for r in grp])
            in_tok = int(statistics.median([r["prompt_tokens"] for r in grp]))
            d = _point_dir(run, s, p)
            tensor = _series_median(d / "dcgm_summary.json", "PIPE_TENSOR_ACTIVE")
            sm = _series_median(d / "dcgm_summary.json", "SM_ACTIVE")
            dram = _series_median(d / "dcgm_summary.json", "DRAM_ACTIVE")
            print(f"  {in_tok:>7} {_fmt(ttft):>9} {_fmt(tensor):>8} {_fmt(sm):>7} {_fmt(dram):>7}")
        print()

    # ── Decode ───────────────────────────────────────────────────────────────
    dec = [(s, p) for (s, p) in by_point if s == "decode" and not p.startswith("_")]
    if dec:
        print("DECODE  (input=1, concurrency=1) — memory-bound regime")
        print(f"  {'out_tok':>7} {'TPOT(ms)':>9} {'dram%':>7} {'tensor%':>8} {'sm%':>7} {'KVcache%':>9}")
        for s, p in sorted(dec, key=lambda x: int(x[1][3:])):
            grp = by_point[(s, p)]
            tpot = _median([r["tpot_ms"] for r in grp])
            out_tok = int(statistics.median([r["completion_tokens"] for r in grp]))
            d = _point_dir(run, s, p)
            dram = _series_median(d / "dcgm_summary.json", "DRAM_ACTIVE")
            tensor = _series_median(d / "dcgm_summary.json", "PIPE_TENSOR_ACTIVE")
            sm = _series_median(d / "dcgm_summary.json", "SM_ACTIVE")
            kv = _series_median(d / "vllm_summary.json", "cache_usage_perc")
            print(f"  {out_tok:>7} {_fmt(tpot):>9} {_fmt(dram):>7} {_fmt(tensor):>8} {_fmt(sm):>7} {_fmt(kv):>9}")
        print()

    print("Note: DCGM PROF_* are fractions 0–1 (×100 for %). Short prefill points")
    print("      have coarse GPU windows; trust their TTFT, not their active%.")


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:g}"


if __name__ == "__main__":
    main()
