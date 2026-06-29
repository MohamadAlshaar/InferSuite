#!/usr/bin/env python3
"""
results_cli.py — GenAI workload characterization terminal results viewer.

New run layout:
  run_TIMESTAMP/
    tok64/   cell_rag_short/  cell_sc_a_short/  cell_sc_b_short/  cell_llm_direct_short/ ...
    tok192/  (latency only)
    tok512/  (latency only)
    tma/
    calibration/
    run_info.json

Usage:
  python3 scripts/results_cli.py --results-dir benchmark_results/run_XXXXXX
  python3 scripts/results_cli.py              # latest run
  python3 scripts/results_cli.py --only t1 t2 f4 f6
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

console = Console() if HAS_RICH else None
SIZES = ["short", "medium", "long", "very_long"]
SIZE_LABELS = {"short": "~20t", "medium": "~100t", "long": "~500t", "very_long": "~2000t"}
UNITS = {"msec"}

# ── perf parsing ──────────────────────────────────────────────────────────────

def parse_perf_totals(path: Path) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or parts[1] == "<not":
            continue
        try:
            val = float(parts[1].replace(",", ""))
        except ValueError:
            continue
        idx = 3 if (len(parts) > 3 and parts[2] in UNITS) else 2
        if idx >= len(parts):
            continue
        event = parts[idx]
        if event.startswith("#") or event.startswith("("):
            continue
        totals[event] = totals.get(event, 0.0) + val
    return totals


def load_cell_perf(cell_dir: Path) -> Dict[str, Dict[str, float]]:
    return {
        "pass1": parse_perf_totals(cell_dir / "perf_pass1.txt"),
        "pass2": parse_perf_totals(cell_dir / "perf_pass2.txt"),
        "pass3": parse_perf_totals(cell_dir / "perf_pass3.txt"),
    }


def derive_hw_metrics(perf: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    p1, p2, p3 = perf.get("pass1",{}), perf.get("pass2",{}), perf.get("pass3",{})
    instr  = p1.get("instructions", 0)
    cycles = p1.get("cycles", 0)
    tc     = p1.get("task-clock", 0)
    brmiss = p1.get("branch-misses", 0)
    l1     = p2.get("L1-dcache-load-misses", 0)
    l2     = p2.get("l2_rqsts.miss", 0)
    llc    = p2.get("cache-misses", 0)
    ref    = p2.get("cache-references", 0)
    avx512 = p3.get("fp_arith_inst_retired.512b_packed_single", 0)
    avx256 = p3.get("fp_arith_inst_retired.256b_packed_single", 0)
    scalar = p3.get("fp_arith_inst_retired.scalar_single", 0)
    amx    = p3.get("exe.amx_busy", 0)

    ipc      = instr / cycles if cycles else 0
    l1_mpki  = l1  / instr * 1000 if instr else 0
    l2_mpki  = l2  / instr * 1000 if instr else 0
    llc_mpki = llc / instr * 1000 if instr else 0
    brmis_ki = brmiss / instr * 1000 if instr else 0
    total_fp = avx512 + avx256 + scalar
    # Suppress AVX% if FP instructions < 0.1% of total (noise from idle CPU)
    fp_ratio = total_fp / instr if instr else 0
    avx512_pct = avx512 / total_fp * 100 if total_fp and fp_ratio > 0.001 else 0.0
    avx256_pct = avx256 / total_fp * 100 if total_fp and fp_ratio > 0.001 else 0.0
    amx_pct    = amx / cycles * 100 if cycles else 0
    bw_gbs     = llc * 64 / (max(tc/1000, 0.001) * 1e9)
    flops      = avx512 * 16 + avx256 * 8 + scalar
    arith_int  = flops / max(llc * 64, 1)

    return {
        "ipc": round(ipc, 2),
        "l1_mpki": round(l1_mpki, 1),
        "l2_mpki": round(l2_mpki, 1),
        "llc_mpki": round(llc_mpki, 2),
        "brmis_ki": round(brmis_ki, 2),
        "avx512_pct": round(avx512_pct, 1),
        "avx256_pct": round(avx256_pct, 1),
        "amx_pct": round(amx_pct, 2),
        "bw_gbs": round(bw_gbs, 2),
        "arith_intensity": round(arith_int, 4),
    }


def parse_tma_slots(path: Path) -> Dict[str, float]:
    """Parse perf stat topdown-* slot counts into percentages."""
    if not path.exists():
        return {}
    totals: Dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3 or parts[1] == "<not":
            continue
        try:
            val = float(parts[1].replace(",", ""))
        except ValueError:
            continue
        idx = 2
        if len(parts) > 3 and parts[2] in UNITS:
            idx = 3
        if idx >= len(parts):
            continue
        event = parts[idx]
        if event.startswith("#") or event.startswith("("):
            continue
        totals[event] = totals.get(event, 0.0) + val

    slots    = totals.get("slots", 0)
    retiring = totals.get("topdown-retiring", 0)
    fe       = totals.get("topdown-fe-bound", 0)
    bad      = totals.get("topdown-bad-spec", 0)
    be       = totals.get("topdown-be-bound", 0)
    total    = retiring + fe + bad + be
    if total == 0:
        return {}
    return {
        "Retiring":      round(retiring / total * 100, 1),
        "Frontend Bound":round(fe       / total * 100, 1),
        "Bad Speculation":round(bad     / total * 100, 1),
        "Backend Bound": round(be       / total * 100, 1),
    }


def parse_tma_toplev(path: Path) -> Dict[str, float]:
    """Parse toplev.py -l2 output."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        m = re.search(
            r'(Backend_Bound(?:\.Memory_Bound|\.Core_Bound)?|'
            r'Frontend_Bound(?:\.Fetch_Latency)?|Retiring|Bad_Speculation)\s+%\s+Slots\s+([\d.]+)',
            line
        )
        if m:
            name_map = {
                "Backend_Bound": "Backend Bound",
                "Backend_Bound.Memory_Bound": "Memory Bound",
                "Backend_Bound.Core_Bound": "Core Bound",
                "Frontend_Bound": "Frontend Bound",
                "Frontend_Bound.Fetch_Latency": "Fetch Latency",
                "Retiring": "Retiring",
                "Bad_Speculation": "Bad Speculation",
            }
            key = name_map.get(m.group(1), m.group(1))
            result[key] = float(m.group(2))
    return result


def load_stream(run_dir: Path) -> Dict[str, float]:
    f = run_dir / "calibration" / "stream.txt"
    result = {}
    if not f.exists():
        return {"Triad": 350.9}
    for line in f.read_text().splitlines():
        for k in ["Copy", "Scale", "Add", "Triad"]:
            if f"{k}:" in line:
                parts = line.split()
                try:
                    result[k] = float(parts[1]) / 1000.0
                except (ValueError, IndexError):
                    pass
    return result


# ── latency CSV helpers ───────────────────────────────────────────────────────

def load_csv(cell_dir: Path) -> List[Dict]:
    csvs = sorted(cell_dir.glob("*.csv"))
    if not csvs:
        return []
    rows = []
    with open(csvs[0]) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return round(s[min(int(len(s) * p / 100), len(s) - 1)], 1)


def _mean(data: List[float]) -> float:
    return round(sum(data) / len(data), 1) if data else 0.0


def _flt(rows: List[Dict], field: str) -> List[float]:
    out = []
    for r in rows:
        try:
            v = float(r.get(field, 0) or 0)
            if v > 0:
                out.append(v)
        except (ValueError, TypeError):
            pass
    return out


def latency_stats(rows: List[Dict]) -> Dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    hits   = [r for r in ok if r.get("cache_hit") == "True"]
    misses = [r for r in ok if r.get("cache_hit") != "True"]
    e2e    = [float(r["e2e_ms"]) for r in ok]
    ttft   = [float(r["frontend_overhead_ms"]) for r in ok]
    gpu    = _flt(ok, "model_backend_http_ms")
    tpot   = [float(r["model_backend_http_ms"]) / float(r["n_output_tokens"])
               for r in ok
               if float(r.get("n_output_tokens") or 0) > 0
               and float(r.get("model_backend_http_ms") or 0) > 0]
    routes = {}
    for r in ok:
        routes[r.get("route", "")] = routes.get(r.get("route", ""), 0) + 1
    return {
        "n": len(ok), "total": len(rows),
        "hits": hits, "misses": misses,
        "e2e": e2e, "ttft": ttft, "gpu": gpu, "tpot": tpot,
        "hit_rate": len(hits) / max(len(ok), 1),
        "rag_embed": _flt(ok, "rag_embed_ms"),
        "rag_milvus": _flt(ok, "rag_milvus_ms"),
        "rag_seaweed": _flt(ok, "rag_seaweed_ms"),
        "rag_format": _flt(ok, "rag_format_ms"),
        "cache_embed": _flt(ok, "cache_embed_ms"),
        "cache_milvus": _flt(ok, "cache_milvus_ms"),
        "cache_mongo": _flt(ok, "cache_mongo_ms"),
        "cache_write": _flt(ok, "cache_write_ms"),
        "n_output_tokens": _flt(ok, "n_output_tokens"),
        "routes": routes,
        "rag_top_score": _flt(ok, "rag_top_score"),
        "rag_num_chunks": _flt(ok, "rag_num_chunks"),
    }


# ── Table 1: Latency Breakdown ────────────────────────────────────────────────

def table1_latency(run_dir: Path) -> None:
    tok_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("tok") and d.is_dir()])
    if not tok_dirs:
        tok_dirs = [run_dir]  # old single-tier layout

    for tok_dir in tok_dirs:
        tok_label = tok_dir.name.replace("tok", "") + " output tokens"
        if HAS_RICH:
            t = Table(title=f"Table 1 — Latency Breakdown — {tok_label}",
                      box=box.ROUNDED, show_header=True, header_style="bold cyan")
            for col, just in [
                ("Path","left"),("Size","left"),("n","right"),
                ("E2E p50","right"),("E2E p95","right"),("TTFT_cpu","right"),
                ("GPU ms","right"),("backend_ms/tok","right"),
                ("Embed ms","right"),("Milvus ms","right"),
                ("Seaweed ms","right"),("CacheWr ms","right"),("Hit%","right"),
            ]:
                t.add_column(col, justify=just)
        else:
            print(f"\n=== Table 1 — {tok_label} ===")

        path_map = [
            ("rag",         "rag",        SIZES),
            ("sc_a",        "sc_a",       ["short","medium"]),
            ("sc_b",        "sc_b",       ["short","medium"]),
            ("llm_direct",  "llm_direct", SIZES),
        ]

        for display, fs_path, sizes in path_map:
            for size in sizes:
                cell_dir = tok_dir / f"cell_{fs_path}_{size}"
                rows = load_csv(cell_dir)
                if not rows:
                    continue
                s = latency_stats(rows)
                ok = [r for r in rows if not r.get("error")]

                def add_row(subset, label, hr_override=None):
                    if not subset:
                        return
                    e2e  = [float(r["e2e_ms"]) for r in subset]
                    ttft = [float(r["frontend_overhead_ms"]) for r in subset]
                    gpu  = _flt(subset, "model_backend_http_ms")
                    tpot = [float(r["model_backend_http_ms"]) / float(r["n_output_tokens"])
                            for r in subset
                            if float(r.get("n_output_tokens") or 0) > 0
                            and float(r.get("model_backend_http_ms") or 0) > 0]
                    emb = [float(r.get("rag_embed_ms",0) or 0) + float(r.get("cache_embed_ms",0) or 0)
                           for r in subset]
                    mil = [float(r.get("rag_milvus_ms",0) or 0) + float(r.get("cache_milvus_ms",0) or 0)
                           for r in subset]
                    sw  = _flt(subset, "rag_seaweed_ms")
                    cw  = _flt(subset, "cache_write_ms")
                    hr  = hr_override if hr_override is not None else s["hit_rate"]
                    hr_str = f"{hr*100:.0f}%" if display in ("sc_a","sc_b") else "—"

                    if HAS_RICH:
                        t.add_row(
                            label, SIZE_LABELS.get(size, size), str(len(subset)),
                            f"{_pct(e2e,50):.0f}", f"{_pct(e2e,95):.0f}",
                            f"{_mean(ttft):.0f}",
                            f"{_mean(gpu):.0f}" if gpu else "—",
                            f"{_mean(tpot):.2f}" if tpot else "—",
                            f"{_mean(emb):.1f}" if any(v>0 for v in emb) else "—",
                            f"{_mean(mil):.1f}" if any(v>0 for v in mil) else "—",
                            f"{_mean(sw):.1f}" if sw else "—",
                            f"{_mean(cw):.1f}" if cw else "—",
                            hr_str,
                        )

                if display in ("sc_a", "sc_b"):
                    hits   = [r for r in ok if r.get("cache_hit") == "True"]
                    misses = [r for r in ok if r.get("cache_hit") != "True"]
                    hr_actual = len(hits) / max(len(ok), 1)
                    add_row(hits,   f"{display}/hit",  hr_actual)
                    if misses:
                        add_row(misses, f"{display}/miss", hr_actual)
                else:
                    add_row(ok, f"{display}/{size[:4]}")

        if HAS_RICH:
            console.print(t)


# ── Table 2: Hardware Counters (tok64 only) ────────────────────────────────────

def table2_hardware(run_dir: Path) -> None:
    tok64 = run_dir / "tok64"
    if not tok64.exists():
        tok64 = run_dir  # fallback

    if HAS_RICH:
        t = Table(title="Table 2 — Hardware Counters (64-token run, all cells)",
                  box=box.ROUNDED, header_style="bold cyan")
        for col in ["Cell","IPC","L1 MPKI","L2 MPKI","LLC MPKI","BrMis/kI","AVX-512%","AMX%","BW GB/s"]:
            t.add_column(col, justify="right" if col != "Cell" else "left")
    else:
        print("\n=== Table 2 — Hardware Counters ===")

    cells = [
        ("rag",       ["short","medium","long","very_long"]),
        ("sc_a",      ["short","medium"]),
        ("sc_b",      ["short","medium"]),
        ("llm_direct",["short","medium","long","very_long"]),
    ]
    for path, sizes in cells:
        for size in sizes:
            cell_dir = tok64 / f"cell_{path}_{size}"
            perf = load_cell_perf(cell_dir)
            if not any(perf.values()):
                continue
            hw = derive_hw_metrics(perf)
            label = f"{path}/{size}"
            if HAS_RICH:
                t.add_row(
                    label,
                    str(hw["ipc"]),
                    str(hw["l1_mpki"]),
                    str(hw["l2_mpki"]),
                    str(hw["llc_mpki"]),
                    str(hw["brmis_ki"]),
                    f"{hw['avx512_pct']:.1f}%",
                    f"{hw['amx_pct']:.2f}%",
                    f"{hw['bw_gbs']:.2f}",
                )
    if HAS_RICH:
        console.print(t)


# ── Figure 1: TMA bottleneck breakdown (slots + toplev) ───────────────────────

def fig1_tma(run_dir: Path) -> None:
    tma_dir = run_dir / "tma"
    if not tma_dir.exists():
        return

    labels, retiring, fe, bad, be = [], [], [], [], []
    for path_mode in ["rag", "llm_direct", "sc_a"]:
        slots = parse_tma_slots(tma_dir / f"tma_slots_{path_mode}.txt")
        if not slots:
            # fallback to toplev
            slots = parse_tma_toplev(tma_dir / f"tma_toplev_{path_mode}.txt")
        if not slots:
            continue
        labels.append(path_mode.replace("sc_a", "SC"))
        retiring.append(slots.get("Retiring", 0))
        fe.append(slots.get("Frontend Bound", 0))
        bad.append(slots.get("Bad Speculation", 0))
        be.append(slots.get("Backend Bound", 0))

    if not labels:
        return

    if HAS_RICH:
        t = Table(title="Figure 1 — TMA: Pipeline Slot Breakdown (%)",
                  box=box.SIMPLE, header_style="bold magenta")
        t.add_column("Metric", style="bold")
        for lbl in labels:
            t.add_column(lbl, justify="right")
        for metric, vals in [
            ("Retiring", retiring), ("Frontend Bound", fe),
            ("Bad Speculation", bad), ("Backend Bound", be),
        ]:
            row = [metric] + [f"{v:.1f}%" if v else "—" for v in vals]
            t.add_row(*row)
        console.print(t)

    if HAS_PLOTEXT:
        plt.clf()
        plt.title("Figure 1 — TMA Pipeline Slot Breakdown")
        for data, lbl in [(retiring,"Retiring"),(fe,"FE Bound"),(bad,"Bad Spec"),(be,"BE Bound")]:
            plt.bar(labels, data, label=lbl)
        plt.ylabel("% of slots")
        plt.show()


# ── Figure 2: Cache miss waterfall ────────────────────────────────────────────

def fig2_waterfall(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return
    tok64 = run_dir / "tok64" if (run_dir / "tok64").exists() else run_dir

    labels, l1, l2, llc = [], [], [], []
    for path, size in [
        ("rag","short"),("rag","medium"),("rag","long"),("rag","very_long"),
        ("sc_a","short"),("sc_a","medium"),
        ("llm_direct","short"),("llm_direct","medium"),("llm_direct","long"),("llm_direct","very_long"),
    ]:
        cell_dir = tok64 / f"cell_{path}_{size}"
        perf = load_cell_perf(cell_dir)
        if not any(perf.values()):
            continue
        hw = derive_hw_metrics(perf)
        labels.append(f"{path[:3]}/{size[:3]}")
        l1.append(hw["l1_mpki"])
        l2.append(hw["l2_mpki"])
        llc.append(hw["llc_mpki"])

    if not labels:
        return
    plt.clf()
    plt.title("Figure 2 — Cache Miss Waterfall (MPKI, all cells)")
    plt.bar(labels, l1, label="L1 MPKI")
    plt.bar(labels, l2, label="L2 MPKI")
    plt.bar(labels, llc, label="LLC MPKI")
    plt.ylabel("Misses per 1K instructions")
    plt.show()


# ── Figure 3: Instruction mix ────────────────────────────────────────────────

def fig3_instmix(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return
    tok64 = run_dir / "tok64" if (run_dir / "tok64").exists() else run_dir

    labels, avx512, avx256, amx = [], [], [], []
    for path, size in [
        ("rag","short"),("rag","medium"),("rag","long"),("rag","very_long"),
        ("sc_a","short"),("sc_a","medium"),
        ("llm_direct","short"),("llm_direct","medium"),
    ]:
        cell_dir = tok64 / f"cell_{path}_{size}"
        perf = load_cell_perf(cell_dir)
        if not any(perf.values()):
            continue
        hw = derive_hw_metrics(perf)
        labels.append(f"{path[:3]}/{size[:3]}")
        avx512.append(hw["avx512_pct"])
        avx256.append(hw["avx256_pct"])
        amx.append(hw["amx_pct"])

    if not labels:
        return
    plt.clf()
    plt.title("Figure 3 — Instruction Mix (% of FP instructions)")
    plt.bar(labels, avx512, label="AVX-512%")
    plt.bar(labels, avx256, label="AVX-256%")
    plt.bar(labels, amx,    label="AMX%")
    plt.ylabel("Percentage")
    plt.show()


# ── Figure 4: Per-phase latency (with SeaweedFS) ─────────────────────────────

def fig4_phases(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return
    tok64 = run_dir / "tok64" if (run_dir / "tok64").exists() else run_dir

    cell_map = [
        ("rag_short","rag","short"), ("rag_medium","rag","medium"),
        ("rag_long","rag","long"),   ("rag_very_long","rag","very_long"),
        ("sc_a_short","sc_a","short"),("sc_a_medium","sc_a","medium"),
        ("sc_b_short","sc_b","short"),("sc_b_medium","sc_b","medium"),
        ("llm_short","llm_direct","short"),("llm_medium","llm_direct","medium"),
        ("llm_long","llm_direct","long"),  ("llm_vl","llm_direct","very_long"),
    ]

    labels, embed_d, milvus_d, seaweed_d, model_d, cwrite_d = [], [], [], [], [], []
    for label, path, size in cell_map:
        cell_dir = tok64 / f"cell_{path}_{size}"
        rows = load_csv(cell_dir)
        if not rows:
            continue
        ok = [r for r in rows if not r.get("error")]
        if not ok:
            continue
        labels.append(label)
        embed_d.append(_mean(_flt(ok,"rag_embed_ms")) + _mean(_flt(ok,"cache_embed_ms")))
        milvus_d.append(_mean(_flt(ok,"rag_milvus_ms")) + _mean(_flt(ok,"cache_milvus_ms")))
        seaweed_d.append(_mean(_flt(ok,"rag_seaweed_ms")))
        model_d.append(_mean(_flt(ok,"model_backend_http_ms")))
        cwrite_d.append(_mean(_flt(ok,"cache_write_ms")))

    if not labels:
        return
    plt.clf()
    plt.title("Figure 4 — Per-phase Latency (mean ms, stacked)")
    plt.bar(labels, embed_d,   label="Embed")
    plt.bar(labels, milvus_d,  label="Milvus")
    plt.bar(labels, seaweed_d, label="SeaweedFS")
    plt.bar(labels, model_d,   label="LLM HTTP")
    plt.bar(labels, cwrite_d,  label="CacheWrite")
    plt.ylabel("ms")
    plt.show()


# ── Figure 5: E2E vs prompt size (all token counts) ───────────────────────────

def fig5_e2e_vs_size(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return

    tok_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("tok") and d.is_dir()])
    if not tok_dirs:
        tok_dirs = [run_dir]

    plt.clf()
    plt.title("Figure 5 — E2E p50 vs Prompt Size × Token Count")
    xs = [1, 2, 3, 4]

    colors = ["blue+", "red+", "green+", "orange+", "cyan+", "magenta+"]
    ci = 0
    for tok_dir in tok_dirs:
        tok = tok_dir.name
        for path, fs_path in [("RAG","rag"), ("LLM","llm_direct")]:
            e2e_vals = []
            for size in SIZES:
                cell_dir = tok_dir / f"cell_{fs_path}_{size}"
                rows = load_csv(cell_dir)
                if not rows:
                    e2e_vals.append(None)
                    continue
                ok = [r for r in rows if not r.get("error")]
                e2e = [float(r["e2e_ms"]) for r in ok]
                e2e_vals.append(_pct(e2e, 50))
            non_none = [v for v in e2e_vals if v is not None]
            if non_none:
                x_valid = [xs[i] for i, v in enumerate(e2e_vals) if v is not None]
                plt.plot(x_valid, non_none, label=f"{path}/{tok}")
            ci += 1

    plt.xlabel("Prompt size (1=short … 4=very_long)")
    plt.ylabel("E2E p50 (ms)")
    plt.show()


# ── Figure 6: Token count comparison ─────────────────────────────────────────

def fig6_token_comparison(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return

    tok_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("tok") and d.is_dir()])
    if len(tok_dirs) < 2:
        return

    # For RAG/medium: compare GPU time and total E2E across token tiers
    labels, gpu_vals, e2e_vals, tpot_vals = [], [], [], []
    for tok_dir in tok_dirs:
        tok = tok_dir.name.replace("tok","")
        cell_dir = tok_dir / "cell_rag_medium"
        rows = load_csv(cell_dir)
        if not rows:
            continue
        ok = [r for r in rows if not r.get("error")]
        if not ok:
            continue
        gpu  = _flt(ok, "model_backend_http_ms")
        e2e  = [float(r["e2e_ms"]) for r in ok]
        tpot = [float(r["model_backend_http_ms"]) / float(r["n_output_tokens"])
                for r in ok
                if float(r.get("n_output_tokens") or 0) > 0
                and float(r.get("model_backend_http_ms") or 0) > 0]
        labels.append(f"{tok}tok")
        gpu_vals.append(_pct(gpu, 50))
        e2e_vals.append(_pct(e2e, 50))
        tpot_vals.append(_mean(tpot))

    if not labels:
        return
    plt.clf()
    plt.title("Figure 6 — Token Count Comparison (RAG/medium, p50)")
    plt.bar(labels, e2e_vals, label="E2E ms")
    plt.bar(labels, gpu_vals, label="GPU ms")
    plt.ylabel("ms")
    plt.show()


# ── Figure 7: SC Scenario A vs B comparison ───────────────────────────────────

def fig7_sc_comparison(run_dir: Path) -> None:
    if not HAS_RICH:
        return
    tok64 = run_dir / "tok64" if (run_dir / "tok64").exists() else run_dir

    t = Table(title="Figure 7 — SC Scenario A (isolated) vs B (full pipeline)",
              box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Scenario", style="bold")
    t.add_column("Size")
    t.add_column("n_hits")
    t.add_column("Hit%", justify="right")
    t.add_column("E2E-hit p50", justify="right")
    t.add_column("TTFT-hit p50", justify="right")
    t.add_column("RAG_embed ms", justify="right")
    t.add_column("SC_embed ms", justify="right")
    t.add_column("Seaweed ms", justify="right")
    t.add_column("LLM saved ms", justify="right")

    for scenario in ["sc_a", "sc_b"]:
        for size in ["short", "medium"]:
            cell_dir = tok64 / f"cell_{scenario}_{size}"
            rows = load_csv(cell_dir)
            if not rows:
                continue
            ok = [r for r in rows if not r.get("error")]
            hits = [r for r in ok if r.get("cache_hit") == "True"]
            misses = [r for r in ok if r.get("cache_hit") != "True"]
            if not hits:
                continue
            hr = len(hits) / max(len(ok), 1)
            e2e_hit  = [float(r["e2e_ms"]) for r in hits]
            ttft_hit = [float(r["frontend_overhead_ms"]) for r in hits]
            rag_emb  = _flt(hits, "rag_embed_ms")
            sc_emb   = _flt(hits, "cache_embed_ms")
            sw       = _flt(hits, "rag_seaweed_ms")
            # LLM saved = E2E of misses - E2E of hits
            e2e_miss = [float(r["e2e_ms"]) for r in misses] if misses else []
            lllm_saved = f"~{_pct(e2e_miss,50)-_pct(e2e_hit,50):.0f}" if e2e_miss else "—"

            t.add_row(
                scenario.upper(),
                SIZE_LABELS.get(size, size),
                str(len(hits)),
                f"{hr*100:.0f}%",
                f"{_pct(e2e_hit,50):.0f}ms",
                f"{_pct(ttft_hit,50):.0f}ms",
                f"{_mean(rag_emb):.1f}ms" if rag_emb else "—",
                f"{_mean(sc_emb):.1f}ms" if sc_emb else "—",
                f"{_mean(sw):.1f}ms" if sw else "—",
                lllm_saved,
            )
    console.print(t)


# ── Figure 8: IPC vs prompt size ─────────────────────────────────────────────

def fig8_ipc_vs_size(run_dir: Path) -> None:
    if not HAS_PLOTEXT:
        return
    tok64 = run_dir / "tok64" if (run_dir / "tok64").exists() else run_dir

    xs = [1, 2, 3, 4]
    plt.clf()
    plt.title("Figure 8 — On-CPU IPC vs Prompt Size")
    for path in ["rag", "llm_direct"]:
        ipc_vals = []
        for size in SIZES:
            cell_dir = tok64 / f"cell_{path}_{size}"
            perf = load_cell_perf(cell_dir)
            if not any(perf.values()):
                ipc_vals.append(None)
                continue
            hw = derive_hw_metrics(perf)
            ipc_vals.append(hw["ipc"])
        valid = [(xs[i], v) for i, v in enumerate(ipc_vals) if v is not None]
        if valid:
            plt.plot([x for x,_ in valid], [v for _,v in valid], label=path)

    plt.xlabel("Prompt size (1=short … 4=very_long)")
    plt.ylabel("Instructions per cycle")
    plt.show()


# ── Figure 9: Output token distribution ──────────────────────────────────────

def fig9_output_tokens(run_dir: Path) -> None:
    if not HAS_RICH:
        return

    t = Table(title="Figure 9 — Output Token Distribution",
              box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Path/Size")
    t.add_column("n", justify="right")
    t.add_column("mean", justify="right")
    t.add_column("p50", justify="right")
    t.add_column("p95", justify="right")
    t.add_column("min", justify="right")
    t.add_column("max", justify="right")

    tok_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("tok") and d.is_dir()])
    if not tok_dirs:
        tok_dirs = [run_dir]

    for tok_dir in tok_dirs:
        tok = tok_dir.name
        for path, sizes in [("rag", SIZES), ("llm_direct", SIZES)]:
            for size in sizes:
                cell_dir = tok_dir / f"cell_{path}_{size}"
                rows = load_csv(cell_dir)
                if not rows:
                    continue
                ok = [r for r in rows if not r.get("error")]
                toks = [int(r["n_output_tokens"]) for r in ok
                        if r.get("n_output_tokens") and str(r["n_output_tokens"]).strip().isdigit()]
                if not toks:
                    continue
                toks_s = sorted(toks)
                n = len(toks_s)
                t.add_row(
                    f"{path[:3]}/{size[:4]} [{tok}]",
                    str(n),
                    f"{sum(toks)/n:.1f}",
                    str(toks_s[n//2]),
                    str(toks_s[min(int(n*0.95), n-1)]),
                    str(min(toks)),
                    str(max(toks)),
                )
    console.print(t)


# ── Run summary ───────────────────────────────────────────────────────────────

def print_summary(run_dir: Path) -> None:
    info_file = run_dir / "run_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())
        if HAS_RICH:
            console.rule(f"[bold cyan]Benchmark Results — {info.get('timestamp','')}")
            console.print(
                f"  Node: {info.get('instance','')}  Model: {info.get('model','')}\n"
                f"  Counts: tok64={info.get('count_64',info.get('count','?'))}  "
                f"tok192={info.get('count_192','?')}  tok512={info.get('count_512','?')}  "
                f"warmup={info.get('warmup','?')}\n"
                f"  SeaweedFS force: {info.get('seaweed_force','?')}"
            )
    stream = load_stream(run_dir)
    if HAS_RICH and stream:
        console.print(f"  STREAM — Copy:{stream.get('Copy',0):.0f}  Scale:{stream.get('Scale',0):.0f}  "
                      f"Add:{stream.get('Add',0):.0f}  [bold green]Triad:{stream.get('Triad',0):.0f} GB/s[/bold green]")

    # Cell completion
    tok_dirs = sorted([d for d in run_dir.iterdir() if d.name.startswith("tok") and d.is_dir()])
    total = complete = 0
    for tok_dir in tok_dirs:
        for path in ["rag","sc_a","sc_b","llm_direct","bge_isolated","hnsw_isolated"]:
            for size in (["short","medium"] if path in ("sc_a","sc_b","bge_isolated","hnsw_isolated") else SIZES):
                total += 1
                if list((tok_dir / f"cell_{path}_{size}").glob("*.csv")):
                    complete += 1
    if HAS_RICH:
        console.print(f"  Cells complete: [bold]{complete}/{total}[/bold]")
    else:
        print(f"  Cells: {complete}/{total}")


# ── Main ─────────────────────────────────────────────────────────────────────

def find_latest(results_dir: Path):
    runs = sorted(results_dir.glob("run_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="GenAI benchmark results viewer")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--only", nargs="+",
                        choices=["summary","t1","t2","f1","f2","f3","f4","f5","f6","f7","f8","f9","all"],
                        default=["all"])
    args = parser.parse_args()

    results_root = Path(__file__).parent.parent / "benchmark_results"
    run_dir = Path(args.results_dir) if args.results_dir else find_latest(results_root)
    if not run_dir or not run_dir.exists():
        print(f"[ERROR] No results found")
        sys.exit(1)

    show = set(args.only)
    all_on = "all" in show

    print_summary(run_dir)
    if all_on or "t1" in show: table1_latency(run_dir)
    if all_on or "t2" in show: table2_hardware(run_dir)
    if all_on or "f1" in show: fig1_tma(run_dir)
    if all_on or "f2" in show: fig2_waterfall(run_dir)
    if all_on or "f3" in show: fig3_instmix(run_dir)
    if all_on or "f4" in show: fig4_phases(run_dir)
    if all_on or "f5" in show: fig5_e2e_vs_size(run_dir)
    if all_on or "f6" in show: fig6_token_comparison(run_dir)
    if all_on or "f7" in show: fig7_sc_comparison(run_dir)
    if all_on or "f8" in show: fig8_ipc_vs_size(run_dir)
    if all_on or "f9" in show: fig9_output_tokens(run_dir)


if __name__ == "__main__":
    main()
