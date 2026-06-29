#!/usr/bin/env python3
"""
generate_report.py — produces benchmark_report.html from a v4 run directory.

Parses pass1, pass2a/2b, pass3 (DRAM cas_count), pass4 (MLP) perf files,
TMA toplev/slots files, and per-request CSVs. Renders a self-contained HTML
report using Chart.js (CDN).

Usage:
  python3 scripts/generate_report.py
  python3 scripts/generate_report.py --run benchmark_results/run_XXXXXX
"""
from __future__ import annotations
import argparse, csv, json, os, re, statistics, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── perf parsing ──────────────────────────────────────────────────────────────

UNITS = {"msec", "MiB", "GiB", "KiB"}

_UNSUPPORTED_SENTINEL = -1.0  # distinct from 0.0 (real measurement of zero)

def parse_perf_totals(path: Path) -> Dict[str, float]:
    """Sum perf stat output per event.

    Returns {event: value} where value == _UNSUPPORTED_SENTINEL (-1.0) means
    the hardware does not support that event (perf reported '<not supported>').
    Callers must treat -1.0 as missing data, not as a count of -1.
    """
    totals: Dict[str, float] = {}
    if not path.exists():
        return {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Detect "<not supported>" — mark with sentinel so callers can warn
        # Handles both formats: standard (VALUE EVENT) and interval (TIMESTAMP VALUE EVENT)
        not_sup_idx = next((i for i, p in enumerate(parts) if p == "<not"), None)
        if not_sup_idx is not None and not_sup_idx + 1 < len(parts) and parts[not_sup_idx + 1] in ("supported>", "counted>"):
            event_idx = not_sup_idx + 2
            if event_idx < len(parts):
                event = parts[event_idx]
                if not (event.startswith("#") or event.startswith("(")):
                    if event not in totals:
                        totals[event] = _UNSUPPORTED_SENTINEL
            continue

        # Auto-detect format:
        #   standard perf stat:    VALUE EVENT [# ...]  → parts[0]=value, parts[1]=event
        #   interval-print format: TIMESTAMP VALUE EVENT → parts[0]=ts, parts[1]=value, parts[2]=event
        # Distinguish by checking if parts[0] looks like a small float timestamp (< 10000, has '.')
        if '.' in parts[0] and len(parts) >= 3:
            try:
                ts = float(parts[0])
                if ts < 10000:  # plausible elapsed-seconds timestamp
                    val = float(parts[1].replace(",", ""))
                    offset = 2
                else:
                    raise ValueError
            except ValueError:
                try:
                    val = float(parts[0].replace(",", ""))
                    offset = 1
                except ValueError:
                    continue
        else:
            try:
                val = float(parts[0].replace(",", ""))
                offset = 1
            except ValueError:
                continue

        if offset >= len(parts):
            continue
        # Skip unit tokens (msec, GHz, etc.) between value and event name
        idx = offset + 1 if (len(parts) > offset + 1 and parts[offset] in UNITS) else offset
        if idx >= len(parts):
            continue
        event = parts[idx]
        if event.startswith("#") or event.startswith("(") or event in ("seconds", "elapsed"):
            continue
        # If we've previously marked this as unsupported but now have a real value, use it
        prev = totals.get(event, 0.0)
        totals[event] = (prev if prev != _UNSUPPORTED_SENTINEL else 0.0) + val
    return totals


def parse_pass3_imc(path: Path) -> Dict[str, float]:
    """Parse pass3 DRAM bandwidth output. Handles two event-name formats:

    Old (pre-SPR): VALUE MiB uncore_imc/cas_count_read/  (MiB already scaled)
    New (SPR/c7i): VALUE       uncore_cha/unc_cha_imc_reads_count.normal/
                   VALUE       uncore_cha/unc_cha_imc_writes_count.full/
                   Each count = 1 cache line = 64 bytes — must convert to MiB.

    Also handles interval-print prefix: TIMESTAMP VALUE UNIT EVENT
    Returns GB/s averages and totals.
    """
    if not path.exists():
        return {}
    total_r = total_w = 0.0
    peak_r = peak_w = 0.0
    intervals = 0
    last_t = 0.0
    elapsed_sec = 0.0
    _CACHE_LINE = 64  # bytes per cache line
    for line in path.read_text(errors="ignore").splitlines():
        rw = None; val = None; mib = None; t = None

        # ── Old format (MiB): optional-timestamp VALUE MiB|GiB uncore_imc/cas_count_(read|write)/ ──
        m = re.search(r'^\s*(?:([\d.]+)\s+)?([\d,.]+)\s+(MiB|GiB|KiB|B)\s+uncore_imc/cas_count_(read|write)/', line)
        if m:
            if m.group(1):
                t = float(m.group(1))
                last_t = max(last_t, t)
            val_raw = float(m.group(2).replace(",", ""))
            unit = m.group(3)
            rw   = m.group(4)
            mib = (val_raw * 1024 if unit == "GiB" else val_raw if unit == "MiB"
                   else val_raw / 1024 if unit == "KiB" else val_raw / 1_048_576)
        else:
            # ── New format (cache-line counts): optional-timestamp VALUE uncore_cha/unc_cha_imc_(reads|writes)... ──
            m2 = re.search(r'^\s*(?:([\d.]+)\s+)?([\d,.]+)\s+uncore_cha/unc_cha_imc_(reads|writes)', line)
            if m2:
                if m2.group(1):
                    t = float(m2.group(1))
                    last_t = max(last_t, t)
                val_raw = float(m2.group(2).replace(",", ""))
                rw = "read" if m2.group(3) == "reads" else "write"
                mib = val_raw * _CACHE_LINE / 1_048_576  # cache lines → MiB
            else:
                es = re.search(r'([\d.]+)\s+seconds time elapsed', line)
                if es:
                    elapsed_sec = float(es.group(1))
                continue

        if rw == "read":
            total_r += mib
            if t is not None:
                intervals += 1
                peak_r = max(peak_r, mib)
        else:
            total_w += mib
            if t is not None:
                peak_w = max(peak_w, mib)
    if not total_r and not total_w:
        return {}
    # Standard format: use elapsed_sec; interval format: use last_t or interval count
    wall = elapsed_sec if elapsed_sec > 0 else max(last_t, intervals * 0.5)
    if wall <= 0:
        return {}
    avg_read_gbs = total_r / 1024.0 / wall
    avg_write_gbs = total_w / 1024.0 / wall
    # For standard format (no per-interval peaks), peak == average
    if intervals == 0:
        peak_r = total_r / wall        # MiB/s
        peak_w = total_w / wall
        peak_r_gbs = round(avg_read_gbs, 3)
        peak_w_gbs = round(avg_write_gbs, 3)
    else:
        peak_r_gbs = round(peak_r * 2 / 1024, 3)
        peak_w_gbs = round(peak_w * 2 / 1024, 3)
    return {
        "intervals": intervals,
        "wall_sec": round(wall, 1),
        "total_read_mib": round(total_r),
        "total_write_mib": round(total_w),
        "peak_read_gbs":  peak_r_gbs,
        "peak_write_gbs": peak_w_gbs,
        "avg_read_gbs":   round(avg_read_gbs, 3),
        "avg_write_gbs":  round(avg_write_gbs, 3),
        "avg_total_gbs":  round(avg_read_gbs + avg_write_gbs, 3),
    }


def parse_pass4_mem(path: Path) -> Dict[str, float]:
    """Parse pass4 mem_load_retired + exe_activity + l1d_pend_miss events.

    Aggregates totals across all interval-print samples.
    Includes new MLP events: l1d_pend_miss.pending / pending_cycles.
    """
    if not path.exists():
        return {}
    keys = {
        "l1_hit": "mem_load_retired.l1_hit",
        "l2_hit": "mem_load_retired.l2_hit",
        "l3_hit": "mem_load_retired.l3_hit",
        "bound_on_loads":   "exe_activity.bound_on_loads",
        "bound_on_stores":  "exe_activity.bound_on_stores",
        "ports_util":       "exe_activity.1_ports_util",
        "l1d_pend":         "l1d_pend_miss.pending",
        "l1d_pend_cycles":  "l1d_pend_miss.pending_cycles",
    }
    sums = {k: 0 for k in keys}
    for line in path.read_text(errors="ignore").splitlines():
        for k, ev in keys.items():
            m = re.search(r'\s([\d,]+)\s+' + re.escape(ev) + r'\b', line)
            if m:
                try:
                    sums[k] += int(m.group(1).replace(",", ""))
                except ValueError:
                    pass
    # Merged pass4 (load-bound + MLP) no longer collects mem_load_retired.*, so the
    # old "l1_hit == 0 → empty" guard would discard valid load-bound/MLP data. Only
    # bail when NOTHING was parsed.
    if not any(sums.values()):
        return {}
    total_loads = sums["l1_hit"] + sums["l2_hit"] + sums["l3_hit"]
    return {
        **sums,
        "l1_pct": round(sums["l1_hit"] / total_loads * 100, 2) if total_loads else 0,
        "l2_pct": round(sums["l2_hit"] / total_loads * 100, 2) if total_loads else 0,
        "l3_pct": round(sums["l3_hit"] / total_loads * 100, 2) if total_loads else 0,
    }


def parse_stream(path: Path) -> Dict[str, float]:
    """Parse STREAM benchmark output → per-kernel best MB/s."""
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r'^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+', line)
        if m:
            out[m.group(1).lower() + "_mbs"] = float(m.group(2))
    if "triad_mbs" in out:
        out["triad_gbs"] = round(out["triad_mbs"] / 1024, 1)
        out["scale_gbs"] = round(out["scale_mbs"] / 1024, 1)
        out["copy_gbs"]  = round(out["copy_mbs"]  / 1024, 1)
        out["add_gbs"]   = round(out["add_mbs"]   / 1024, 1)
    return out


def parse_vllm_metrics(path: Path) -> Dict[str, Any]:
    """Read the vllm_metrics_summary.json written by vllm_metrics_scraper.py.

    Shape: {"gauges": {metric: {min,median,max,last,n}},
            "counters": {metric: {start,end,delta,rate_per_s}}, ...}
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def parse_vllm_interval(path: Path) -> List[Dict[str, Any]]:
    """Read perf_pass1_vllm_interval.txt — perf stat -I 200 output.
    Returns list of {t_ms, cycles, instructions, task_clock_ms} dicts, one per interval."""
    if not path.exists():
        return []
    rows = []
    current_t = None
    counters: Dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        # Interval header: " 0.200000000 ..."
        m = re.match(r'^\s*([\d.]+)\s+(\S+)\s+(\S+)', line)
        if not m:
            continue
        try:
            t = float(m.group(1))
            val_str = m.group(2).replace(',', '')
            event = m.group(3)
            val = float(val_str)
        except (ValueError, AttributeError):
            continue
        if current_t is None or abs(t - current_t) > 0.01:
            if current_t is not None and counters:
                rows.append({'t_ms': round(current_t * 1000), **counters})
            current_t = t
            counters = {}
        if 'task-clock' in event:
            counters['task_clock_ms'] = val
        elif 'cycles' in event:
            counters['cycles'] = val
        elif 'instructions' in event:
            counters['instructions'] = val
    if current_t is not None and counters:
        rows.append({'t_ms': round(current_t * 1000), **counters})
    return rows


def parse_gpu_dmon(path: Path) -> Dict[str, float]:
    """Parse nvidia-smi dmon output → average GPU metrics during measurement window.

    Handles both basic -s u columns (sm, mem) and GPM metrics (smutil, smocc, hmmat, dram).
    Values of '-' (unsupported or GPU idle with no data) are skipped.
    Returns averages over all valid samples.
    """
    if not path.exists():
        return {}
    lines = path.read_text(errors="ignore").splitlines()
    # Find header line (starts with '# gpu')
    headers: List[str] = []
    data_lines: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# gpu"):
            headers = stripped.lstrip("# ").split()
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts and parts[0].isdigit() and headers:
            data_lines.append(parts)
    if not headers or not data_lines:
        return {}

    def _col_avg(col: str) -> Optional[float]:
        try:
            idx = headers.index(col)
        except ValueError:
            return None
        vals = []
        for row in data_lines:
            if idx < len(row) and row[idx] not in ("-", "N/A"):
                try:
                    vals.append(float(row[idx]))
                except ValueError:
                    pass
        return round(statistics.mean(vals), 1) if vals else None

    out: Dict[str, float] = {}
    for key, col in [("sm_pct", "sm"), ("mem_pct", "mem"),
                     ("sm_activity_pct", "smutil"), ("sm_occupancy_pct", "smocc"),
                     ("hmma_tensor_pct", "hmmat"), ("dram_activity_pct", "dram")]:
        v = _col_avg(col)
        if v is not None:
            out[key] = v
    return out


def parse_gpu_pmon(path: Path) -> Dict[str, Any]:
    """Parse nvidia-smi pmon output → per-compute-process GPU metrics.

    Returns dict keyed by PID with sm%, mem%, fb_mb, and command name.
    Only includes Compute (C) processes — excludes display (G) processes.
    """
    if not path.exists():
        return {}
    lines = path.read_text(errors="ignore").splitlines()
    headers: List[str] = []
    pid_data: Dict[str, Dict[str, List]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# gpu"):
            headers = stripped.lstrip("# ").split()
            continue
        if stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or not parts[0].isdigit() or not headers:
            continue
        try:
            pid_idx = headers.index("pid")
            type_idx = headers.index("type")
            pid = parts[pid_idx] if pid_idx < len(parts) else ""
            ptype = parts[type_idx] if type_idx < len(parts) else ""
        except (ValueError, IndexError):
            continue
        if ptype != "C":
            continue
        if pid not in pid_data:
            cmd_idx = len(headers) - 1
            cmd = parts[cmd_idx] if cmd_idx < len(parts) else "unknown"
            pid_data[pid] = {"sm": [], "mem": [], "fb": [], "command": cmd}
        for field, col in [("sm", "sm"), ("mem", "mem"), ("fb", "fb")]:
            try:
                idx = headers.index(col)
                val = parts[idx] if idx < len(parts) else "-"
                if val not in ("-", "N/A"):
                    pid_data[pid][field].append(float(val))
            except (ValueError, IndexError):
                pass

    result: Dict[str, Any] = {}
    for pid, d in pid_data.items():
        result[pid] = {
            "command": d["command"],
            "sm_pct":  round(statistics.mean(d["sm"]),  1) if d["sm"]  else None,
            "mem_pct": round(statistics.mean(d["mem"]), 1) if d["mem"] else None,
            "fb_mb":   round(statistics.mean(d["fb"]),  0) if d["fb"]  else None,
        }
    return result


def parse_tma_slots(path: Path) -> Dict[str, float]:
    """Parse perf-stat topdown-* slot counts → percent of slots.
    Returns {"_failed": "<reason>"} if the perf invocation died before producing data."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    # Detect known failure modes
    if "command terminated with exit code" in text:
        return {"_failed": "perf command terminated with non-zero exit (kubectl exec failure mid-run)"}
    if "Ignored open failure" in text and "topdown-" not in text:
        return {"_failed": "perf could not attach to all threads (kept warnings only, no measurement)"}
    raw: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            v = float(parts[0].replace(",", ""))
        except ValueError:
            continue
        for p in parts[1:]:
            if p in ("slots", "topdown-retiring", "topdown-fe-bound",
                     "topdown-bad-spec", "topdown-be-bound"):
                raw[p] = raw.get(p, 0.0) + v
                break
    slots = raw.get("slots", 0)
    if slots <= 0:
        return {}
    return {
        "retiring":   round(raw.get("topdown-retiring", 0)  / slots * 100, 1),
        "fe_bound":   round(raw.get("topdown-fe-bound", 0)  / slots * 100, 1),
        "bad_spec":   round(raw.get("topdown-bad-spec", 0)  / slots * 100, 1),
        "be_bound":   round(raw.get("topdown-be-bound", 0)  / slots * 100, 1),
    }


def parse_tma_toplev(path: Path) -> Dict[str, float]:
    """Parse toplev -l1 or -l2 output → level-1 and level-2 percentages.

    Level-1: be_bound, fe_bound, retiring, bad_spec
    Level-2: memory_bound, core_bound, l1_bound, l2_bound, l3_bound,
             dram_bound, store_bound, ports_utilization, divider

    Returns {"_failed": "..."} if toplev died before producing output.
    """
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    if "command terminated" in text:
        return {"_failed": "toplev/perf command terminated mid-run"}
    out: Dict[str, float] = {}

    # Level-2 node name → output key mapping
    L2_MAP = {
        "Memory_Bound":      "memory_bound",
        "Core_Bound":        "core_bound",
        "L1_Bound":          "l1_bound",
        "L2_Bound":          "l2_bound",
        "L3_Bound":          "l3_bound",
        "DRAM_Bound":        "dram_bound",
        "Store_Bound":       "store_bound",
        "Ports_Utilization": "ports_utilization",
        "Divider":           "divider",
        "Frontend_Bound":    "fe_bound",
        "Bad_Speculation":   "bad_spec",
        "Backend_Bound":     "be_bound",
        "Retiring":          "retiring",
    }

    for line in text.splitlines():
        # Actual toplev output format (with or without leading whitespace):
        #   "BE               Backend_Bound                    % Slots   66.0  [50.0%]"
        #   "BE/Mem           Backend_Bound.Memory_Bound       % Slots   24.0  [50.0%]"
        #   "BE/Mem           Backend_Bound.Memory_Bound.L3_Bound  % Stalls  11.5  [50.0%]"
        #   "BE/Core          Backend_Bound.Core_Bound.Ports_Utilization  % Clocks  15.1"
        # Pattern: category  dotted.node.name  %  Metric  value
        m = re.match(
            r'^\s*(\w+(?:/\w+)?)\s+(\S+)\s+%\s+(?:Slots|Stalls|Clocks)\s+([\d.]+)',
            line,
        )
        if not m:
            continue
        # Use the last component of the dotted node name for lookup
        node_name = m.group(2).split(".")[-1]
        val = float(m.group(3))
        key = L2_MAP.get(node_name)
        if key:
            out[key] = val

    return out


# ── TMA narrative interpretation ──────────────────────────────────────────────

# TMA headline labels only — no speculation. The companion metrics table
# shown alongside each card provides the factual confirmation.
TMA_NARRATIVES: Dict[str, str] = {
    "dram_bound":        "DRAM-Bound: stalls waiting on LLC misses resolved from main memory (~200 cycle penalty per miss on SPR)",
    "l3_bound":          "L3-Bound: L2 misses resolved in LLC (~40 cycle penalty). Working set fits in LLC but not L2.",
    "l2_bound":          "L2-Bound: L1 misses resolved in L2 (~12 cycle penalty). Hot working set fits in L2.",
    "l1_bound":          "L1-Bound: structural L1 hazards (4K aliasing, load-hit-store). Rare unless high thread contention.",
    "store_bound":       "Store-Bound: store buffer full, blocking younger loads. High write traffic is saturating the store buffer.",
    "core_bound":        "Core-Bound: execution units are the bottleneck. Data arrives on time but ALU/FPU ports are saturated.",
    "ports_utilization": "Ports-Utilization: multiple execution ports saturated simultaneously. Compute throughput at instruction-mix ceiling.",
    "divider":           "Divider: long-latency division/sqrt instructions blocking retirement (~20–90 cycles each on SPR).",
    "fe_bound":          "Frontend-Bound: instruction fetch or decode is the bottleneck. Pipeline has empty stages before execution.",
    "bad_spec":          "Bad-Speculation: pipeline work flushed due to branch misprediction or machine clear (~20 wasted cycles per flush).",
}

_TMA_L2_NODES = ("dram_bound", "l3_bound", "l2_bound", "l1_bound", "store_bound",
                  "core_bound", "ports_utilization", "divider", "fe_bound", "bad_spec")

_TMA_L2_NODES = ("dram_bound", "l3_bound", "l2_bound", "l1_bound", "store_bound",
                  "core_bound", "ports_utilization", "divider", "fe_bound", "bad_spec")


def generate_tma_narrative(tma: Dict[str, float], pod_key: str, mode: str,
                           hw: Dict) -> str:
    """Generate data-only TMA card content: bottleneck classifications + companion metrics table."""
    # Per-pod TMA dicts store toplev keys with a "toplev_" prefix (set at load time
    # in load_run() line 1410). Fall back to the prefixed key so per-pod cards render.
    scored = [(v, n) for n in _TMA_L2_NODES
              if (v := tma.get(n, tma.get(f"toplev_{n}"))) is not None and v > 5.0]
    scored.sort(reverse=True)

    if not scored:
        return '<span style="color:var(--text-dim);font-size:11px">No TMA data — toplev did not capture samples for this pod.</span>'

    # ── Bottleneck list ────────────────────────────────────────────────────────
    parts = []
    colors = ["#f85149", "#d29922", "#58a6ff"]
    for rank, (val, node) in enumerate(scored[:4]):
        label = TMA_NARRATIVES.get(node, node)
        color = colors[min(rank, len(colors)-1)]
        parts.append(
            f'<div style="margin:4px 0;font-size:11px">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{color};margin-right:5px"></span>'
            f'<strong>{val:.1f}%</strong> {label}</div>'
        )

    # ── Companion metrics table ────────────────────────────────────────────────
    def _cell(v, fmt="{:.2f}", suffix=""):
        return f"{fmt.format(v)}{suffix}" if v is not None else "—"

    metrics = [
        # ── Execution efficiency ──────────────────────────────────────────────
        ("IPC",              hw.get("ipc"),               "{:.2f}", "",   "Instructions retired per cycle. Fleet avg ~0.60 (Kanev et al.). Low = stalled pipeline."),
        ("ILP",              hw.get("ilp"),               "{:.2f}", "",   "Executed uops per cycle (execution-unit throughput, ignoring stalls)."),
        ("br_mispredict%",   hw.get("br_mispredict_pct"), "{:.1f}", "%",  "Branch misprediction rate. Confirms Bad-Speculation when elevated."),
        ("brmis/Ki",         hw.get("brmis_ki"),          "{:.2f}", "",   "Branch misses per 1000 instructions (Bad-Spec density)."),
        # ── Memory hierarchy ─────────────────────────────────────────────────
        ("L1 MPKI",          hw.get("l1_mpki"),           "{:.1f}", "",   "L1D misses per 1000 instructions."),
        ("L2 MPKI",          hw.get("l2_mpki"),           "{:.1f}", "",   "L2 misses per 1000 instructions."),
        ("LLC MPKI",         hw.get("llc_mpki"),          "{:.2f}", "",   "LLC (L3) misses per 1000 instructions. Confirms DRAM-Bound when high."),
        ("MLP",              hw.get("mlp"),               "{:.1f}", "",   "Avg outstanding L1D misses (memory-level parallelism). <2 = serial stalls; >4 = prefetcher active."),
        ("AMAT* (est)",      hw.get("amat_cycles"),       "{:.1f}", " cy","Estimated AMAT using nominal SPR latencies: L1×4 + L2×12 + L3×40 + DRAM×200 cy. Not directly measured — see eff_llc_lat for a real measured proxy."),
        ("eff LLC lat",      hw.get("eff_llc_lat_cy"),    "{:.0f}", " cy","Effective cycles stalled per LLC miss (stall_cycles / llc_misses). Measures actual DRAM penalty under load."),
        ("DTLB MPKI",        hw.get("dtlb_mpki"),         "{:.2f}", "",   "Data TLB misses per 1000 instructions. Elevated DTLB → page-walk stalls count as Memory-Bound."),
        # ── Frontend pressure ─────────────────────────────────────────────────
        ("ITLB MPKI",        hw.get("itlb_mpki"),         "{:.2f}", "",   "Instruction TLB misses per 1000 instructions. Key driver of Frontend-Bound (large code footprint)."),
        ("ICACHE MPKI",      hw.get("icache_mpki"),       "{:.2f}", "",   "I-cache misses per 1000 instructions. Directly causes instruction-fetch stalls → Frontend-Bound."),
        # ── Stall breakdown ───────────────────────────────────────────────────
        ("mem stall%",       hw.get("mem_stall_pct"),     "{:.1f}", "%",  "% of cycles with a pending load outstanding. Confirms Memory-Bound."),
        ("stalls tot%",      hw.get("stalls_tot_pct"),    "{:.1f}", "%",  "% of cycles where no uops executed. Confirms Backend-Bound."),
    ]

    tbl = ['<table style="width:100%;font-size:10px;border-collapse:collapse;margin-top:6px">']
    for label, val, fmt, suffix, tooltip in metrics:
        if val is None:
            continue
        tbl.append(
            f'<tr title="{tooltip}">'
            f'<td style="color:var(--text-dim);padding:1px 6px 1px 0;white-space:nowrap">{label}</td>'
            f'<td style="font-variant-numeric:tabular-nums;font-weight:600">{_cell(val, fmt, suffix)}</td>'
            f'</tr>'
        )
    tbl.append('</table>')

    parts.append("".join(tbl))
    return "\n".join(parts)


# ── derived hw metrics ────────────────────────────────────────────────────────

def _ratio(num, den, scale=1.0, ndigits=2):
    """Safe ratio. Returns None when the denominator is missing/zero OR when either
    operand is the unsupported sentinel (-1.0), which means the hardware doesn't
    support that event. A genuine 0 numerator with a real denominator returns 0.0."""
    if num is None or den is None or den == 0:
        return None
    if num == _UNSUPPORTED_SENTINEL or den == _UNSUPPORTED_SENTINEL:
        return None
    return round(num / den * scale, ndigits)


def _is_supported(val: float) -> bool:
    return val != _UNSUPPORTED_SENTINEL


def derive_hw(p1: Dict, p2: Dict, p3: Dict, p4: Dict, p5: Dict) -> Dict[str, Any]:
    """Derive hardware metrics. Any metric whose source pass is missing is None,
    rendered as "—" in tables. Never silently substitutes 0 for missing data.
    Events reported as <not supported> by perf use sentinel -1.0 and produce None metrics."""
    out: Dict[str, Any] = {
        "passes_present": {
            "p1": bool(p1), "p2": bool(p2), "p3": bool(p3),
            "p4": bool(p4), "p5": bool(p5),
        },
        "unsupported_events": [],
    }

    # ── pass1: IPC, ILP, branch-miss rate, ctx switches, speculation, frequency ──
    if p1:
        instr    = p1.get("instructions", 0)
        cycles   = p1.get("cycles", 0)
        tc_ms    = p1.get("task-clock", 0)
        brmiss   = p1.get("branch-misses", 0)
        uops_iss = p1.get("uops_issued.any", 0)
        uops_ret = p1.get("uops_retired.slots", 0)
        uops_exe = p1.get("uops_executed.core", 0)

        # Track unsupported events
        for ev, val in p1.items():
            if val == _UNSUPPORTED_SENTINEL:
                out["unsupported_events"].append(f"pass1:{ev}")

        out["instr"]   = int(instr) if _is_supported(instr) else None
        out["cycles"]  = int(cycles) if _is_supported(cycles) else None
        out["task_clock_sec"] = round(tc_ms / 1000, 1) if (tc_ms and _is_supported(tc_ms)) else None
        out["ipc"]            = _ratio(instr, cycles)
        out["ilp"]            = _ratio(uops_exe, cycles)  # avg uops executed per cycle
        out["brmis_ki"]       = _ratio(brmiss, instr, scale=1000, ndigits=2)
        br_total = p1.get("branch-instructions")
        out["br_mispredict_pct"] = _ratio(brmiss, br_total, scale=100, ndigits=1) if (br_total and _is_supported(br_total)) else None
        out["spec_waste_pct"] = (_ratio(uops_iss - uops_ret, uops_iss, scale=100, ndigits=1)
                                 if (uops_iss and _is_supported(uops_iss) and _is_supported(uops_ret)) else None)
        out["ctx_switches"]   = int(p1.get("context-switches", 0)) if ("context-switches" in p1 and _is_supported(p1.get("context-switches", 0))) else None
        out["cpu_migrations"] = int(p1.get("cpu-migrations", 0))   if ("cpu-migrations" in p1 and _is_supported(p1.get("cpu-migrations", 0))) else None
        out["page_faults"]    = int(p1.get("page-faults", 0))      if ("page-faults"      in p1 and _is_supported(p1.get("page-faults", 0)))      else None
        out["avg_freq_ghz"]   = _ratio(cycles, tc_ms * 1e6, ndigits=2) if (tc_ms and _is_supported(tc_ms) and _is_supported(cycles)) else None
    else:
        cycles = 0
        for k in ("instr","cycles","task_clock_sec","ipc","ilp","brmis_ki","spec_waste_pct",
                  "ctx_switches","cpu_migrations","page_faults","avg_freq_ghz"):
            out[k] = None

    # ── pass2: cache hierarchy, stalls ──
    if p2:
        instr_p2 = p1.get("instructions", 0) if p1 else 0
        l1   = p2.get("L1-dcache-load-misses", 0)
        l2   = p2.get("l2_rqsts.miss", 0)
        llc  = p2.get("cache-misses", 0)
        cache_refs = p2.get("cache-references", 0)
        out["l1_mpki"]  = _ratio(l1, instr_p2, scale=1000, ndigits=1) if instr_p2 else None
        out["l2_mpki"]  = _ratio(l2, instr_p2, scale=1000, ndigits=1) if instr_p2 else None
        out["llc_mpki"] = _ratio(llc, instr_p2, scale=1000, ndigits=2) if instr_p2 else None
        out["llc_miss_rate_pct"] = _ratio(llc, cache_refs, scale=100, ndigits=1) if cache_refs else None
        stalls_l3   = p2.get("cycle_activity.stalls_l3_miss")
        stalls_tot  = p2.get("cycle_activity.stalls_total")
        # Prefer within-pass cycles when available (pass2 now records cycles); fall back to pass1
        cycles_p2 = p2.get("cycles") if (p2.get("cycles") and _is_supported(p2.get("cycles"))) else cycles
        out["stalls_tot_pct"]  = _ratio(stalls_tot, cycles_p2, scale=100, ndigits=1) if (stalls_tot is not None and cycles_p2) else None
        out["stalls_l3_pct"]   = _ratio(stalls_l3,  cycles_p2, scale=100, ndigits=1) if (stalls_l3  is not None and cycles_p2) else None
        out["stalls_l3_share"] = _ratio(stalls_l3, stalls_tot, scale=100, ndigits=1) if (stalls_l3 is not None and stalls_tot) else None
        # Kanev et al. WSC metrics: ITLB/ICACHE/DTLB MPKI explain Frontend_Bound and Memory_Bound
        itlb  = p2.get("iTLB-load-misses")
        icach = p2.get("L1-icache-load-misses")
        dtlb  = p2.get("dTLB-load-misses")
        out["itlb_mpki"]  = _ratio(itlb,  instr_p2, scale=1000, ndigits=2) if (itlb  is not None and instr_p2) else None
        out["icache_mpki"]= _ratio(icach, instr_p2, scale=1000, ndigits=2) if (icach is not None and instr_p2) else None
        out["dtlb_mpki"]  = _ratio(dtlb,  instr_p2, scale=1000, ndigits=2) if (dtlb  is not None and instr_p2) else None
        # Effective LLC-miss latency: stall cycles per LLC miss — measures actual DRAM penalty in context
        out["eff_llc_lat_cy"] = round(stalls_tot / llc, 1) if (stalls_tot and llc and llc > 0) else None
    else:
        for k in ("l1_mpki","l2_mpki","llc_mpki","llc_miss_rate_pct",
                  "stalls_tot_pct","stalls_l3_pct","stalls_l3_share",
                  "itlb_mpki","icache_mpki","dtlb_mpki","eff_llc_lat_cy"):
            out[k] = None

    # ── SIMD FP32 mix (pass permanently dropped — p3 is always {}, block never executes) ──
    if p3:
        # FP32
        avx512_fp32 = p3.get("fp_arith_inst_retired.512b_packed_single", 0)
        avx256_fp32 = p3.get("fp_arith_inst_retired.256b_packed_single", 0)
        scalar_fp32 = p3.get("fp_arith_inst_retired.scalar_single", 0)
        # FP64
        avx512_fp64 = p3.get("fp_arith_inst_retired.512b_packed_double", 0)
        avx256_fp64 = p3.get("fp_arith_inst_retired.256b_packed_double", 0)
        scalar_fp64 = p3.get("fp_arith_inst_retired.scalar_double", 0)
        # BF16 (Sapphire Rapids native)
        avx512_bf16 = p3.get("fp_arith_inst_retired.512b_packed_bf16", 0)

        amx       = p3.get("exe.amx_busy", 0)
        hwpf_miss = p3.get("l2_rqsts.hwpf_miss", 0)
        hwpf_all  = p3.get("l2_rqsts.all_hwpf", 0)

        # Coerce unsupported events to 0 for denominator purposes
        def _safe(v): return 0 if (v is None or v == _UNSUPPORTED_SENTINEL) else v

        # Track unsupported events
        for ev, val in p3.items():
            if val == _UNSUPPORTED_SENTINEL:
                out["unsupported_events"].append(f"pass3:{ev}")

        # Total FP denominator includes FP32 + FP64 + BF16
        total_fp = (_safe(avx512_fp32) + _safe(avx256_fp32) + _safe(scalar_fp32) +
                    _safe(avx512_fp64) + _safe(avx256_fp64) + _safe(scalar_fp64) +
                    _safe(avx512_bf16))

        instr_p3 = p1.get("instructions", 0) if p1 else 0
        instr_p3 = 0 if (instr_p3 == _UNSUPPORTED_SENTINEL) else instr_p3
        fp_active = total_fp and instr_p3 and (total_fp / instr_p3) > 0.001

        # Per-precision percentages of all FP ops
        out["avx512_pct"]      = _ratio(_safe(avx512_fp32), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx256_pct"]      = _ratio(_safe(avx256_fp32), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx512_fp32_pct"] = _ratio(_safe(avx512_fp32), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx256_fp32_pct"] = _ratio(_safe(avx256_fp32), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx512_fp64_pct"] = _ratio(_safe(avx512_fp64), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx256_fp64_pct"] = _ratio(_safe(avx256_fp64), total_fp, scale=100, ndigits=1) if fp_active else None
        out["avx512_bf16_pct"] = _ratio(_safe(avx512_bf16), total_fp, scale=100, ndigits=1) if fp_active else None
        out["amx_pct"]         = _ratio(_safe(amx), cycles, scale=100, ndigits=2) if cycles else None

        # FLOPS — corrected for FMA (2 FLOPs per element):
        # AVX-512 FMA FP32: 16 elements × 2 = 32 FLOPs/instruction
        # AVX-256 FMA FP32:  8 elements × 2 = 16 FLOPs/instruction
        # Scalar FP32:                         1 FLOP/instruction
        # AVX-512 FMA FP64:  8 elements × 2 = 16 FLOPs/instruction
        # AVX-256 FMA FP64:  4 elements × 2 =  8 FLOPs/instruction
        # Scalar FP64:                          1 FLOP/instruction
        # AVX-512 BF16:     32 elements × 2 = 32 FLOPs/instruction (SPR VNNI-style)
        if fp_active:
            out["flops"] = int(
                _safe(avx512_fp32) * 32 + _safe(avx256_fp32) * 16 + _safe(scalar_fp32) +
                _safe(avx512_fp64) * 16 + _safe(avx256_fp64) * 8  + _safe(scalar_fp64) +
                _safe(avx512_bf16) * 32
            )
        else:
            out["flops"] = None
    else:
        for k in ("avx512_pct","avx256_pct","avx512_fp32_pct","avx256_fp32_pct",
                  "avx512_fp64_pct","avx256_fp64_pct","avx512_bf16_pct",
                  "amx_pct","flops"):
            out[k] = None

    # ── pass3: real DRAM bandwidth (cas_count_read/write) ──
    p4_keys = ("dram_avg_total_gbs","dram_avg_read_gbs","dram_avg_write_gbs",
               "dram_peak_read_gbs","dram_peak_write_gbs",
               "dram_total_read_gib","dram_total_write_gib","dram_wall_sec",
               "dram_rw_ratio","dram_burst_factor","dram_intervals")
    if p4:
        out["dram_avg_total_gbs"]   = p4.get("avg_total_gbs")
        out["dram_avg_read_gbs"]    = p4.get("avg_read_gbs")
        out["dram_avg_write_gbs"]   = p4.get("avg_write_gbs")
        out["dram_peak_read_gbs"]   = p4.get("peak_read_gbs")
        out["dram_peak_write_gbs"]  = p4.get("peak_write_gbs")
        out["dram_intervals"]       = p4.get("intervals", 0)
        tr = p4.get("total_read_mib") ; tw = p4.get("total_write_mib")
        out["dram_total_read_gib"]  = round(tr / 1024, 2) if tr is not None else None
        out["dram_total_write_gib"] = round(tw / 1024, 2) if tw is not None else None
        out["dram_wall_sec"]        = p4.get("wall_sec")
        out["dram_rw_ratio"]        = _ratio(tr, tw, ndigits=1)
        avg = p4.get("avg_total_gbs")
        peak = (p4.get("peak_read_gbs") or 0) + (p4.get("peak_write_gbs") or 0) if (p4.get("peak_read_gbs") is not None) else None
        out["dram_burst_factor"]    = _ratio(peak, avg, ndigits=1) if (peak and avg and p4.get("intervals",0) > 0) else None
    else:
        for k in p4_keys: out[k] = None

    # Arithmetic intensity — flops per byte of DRAM traffic.
    # SIMD pass was dropped so flops is always None → arith_intensity_fp32 is always None.
    # Kept in derive_hw for backward compat with old results that had pass3 SIMD data.
    flops = out.get("flops")
    if flops is not None and out.get("dram_total_read_gib") is not None:
        total_gib = (out.get("dram_total_read_gib") or 0) + (out.get("dram_total_write_gib") or 0)
        bytes_total = total_gib * (1024 ** 3)
        out["arith_intensity_fp32"] = round(flops / bytes_total, 4) if bytes_total else None
    else:
        out["arith_intensity_fp32"] = None

    # cpu_util_pct REMOVED — would require dividing pass1 task-clock by pass3 (DRAM) wall,
    # a cross-pass ratio with the same flaw we already flag for stalls_tot_pct.
    # If we want it correctly, add task-clock as an event in pass3 and divide
    # within that pass.

    # ── pass4: memory pyramid, exe_activity stalls, MLP, AMAT ──
    p5_keys = ("l1_hits","l2_hits","l3_hits","l1_pct","l2_pct","l3_pct",
               "bound_on_loads","bound_on_stores","ports_util",
               "mem_stall_pct","store_stall_pct","ports_util_pct",
               "mlp","amat_cycles")
    if p5:
        out["l1_hits"] = p5.get("l1_hit")
        out["l2_hits"] = p5.get("l2_hit")
        out["l3_hits"] = p5.get("l3_hit")
        out["l1_pct"]  = p5.get("l1_pct")
        out["l2_pct"]  = p5.get("l2_pct")
        out["l3_pct"]  = p5.get("l3_pct")
        out["bound_on_loads"]  = p5.get("bound_on_loads")
        out["bound_on_stores"] = p5.get("bound_on_stores")
        out["ports_util"]      = p5.get("ports_util")
        # Prefer within-pass cycles when available (pass4 now records cycles); fall back to pass1
        cycles_p5 = p5.get("cycles") if (p5.get("cycles") and _is_supported(p5.get("cycles"))) else cycles
        _msp = _ratio(p5.get("bound_on_loads"), cycles_p5, scale=100, ndigits=1) if cycles_p5 else None
        # Cap at 100% — values over 100 indicate counter multiplexing artifacts
        out["mem_stall_pct"] = min(_msp, 100.0) if _msp is not None else None
        out["store_stall_pct"] = _ratio(p5.get("bound_on_stores"), cycles_p5, scale=100, ndigits=1) if cycles_p5 else None
        out["ports_util_pct"]  = _ratio(p5.get("ports_util"),      cycles_p5, scale=100, ndigits=1) if cycles_p5 else None

        # MLP = avg outstanding L1D misses per cycle when there is at least one miss
        # MLP ~1.0 = serial stalls; MLP ~8+ = parallel / prefetch-assisted
        pend      = p5.get("l1d_pend", 0) or 0
        pend_cyc  = p5.get("l1d_pend_cycles", 0) or 0
        out["mlp"] = _ratio(pend, pend_cyc, ndigits=2) if (pend and pend_cyc) else None

        # AMAT in cycles using fixed Sapphire Rapids latencies: L1=4, L2=12, L3=40, DRAM=200
        l1h = p5.get("l1_hit", 0) or 0
        l2h = p5.get("l2_hit", 0) or 0
        l3h = p5.get("l3_hit", 0) or 0
        total_loads_p5 = l1h + l2h + l3h
        if total_loads_p5 > 0:
            llc_mpki = out.get("llc_mpki") or 0
            llc_miss_p5 = max(0, total_loads_p5 * llc_mpki / 1000) if llc_mpki else 0
            amat_num = l1h * 4 + l2h * 12 + l3h * 40 + llc_miss_p5 * 200
            out["amat_cycles"] = round(amat_num / total_loads_p5, 2)
        else:
            out["amat_cycles"] = None
    else:
        for k in p5_keys: out[k] = None

    return out


# ── csv loading ───────────────────────────────────────────────────────────────

def _f(s: str) -> float:
    try: return float(s)
    except (ValueError, TypeError): return 0.0

def _i(s: str) -> int:
    try: return int(s)
    except (ValueError, TypeError): return 0

def _pct(values: List[float], q: float) -> float:
    if not values: return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]

def latency_stats_main(rows: List[Dict]) -> Dict[str, Any]:
    """Stats from a main-pipeline cell CSV (rag/sc_a/sc_b/llm_direct)."""
    if not rows:
        return {}
    e2e = [_f(r.get("e2e_ms")) for r in rows if r.get("e2e_ms")]
    mb  = [_f(r.get("model_backend_http_ms")) for r in rows]
    ttft = [_f(r.get("frontend_overhead_ms")) for r in rows]
    nout = [_i(r.get("n_output_tokens")) for r in rows]
    embed = [_f(r.get("rag_embed_ms")) for r in rows]
    milvus = [_f(r.get("rag_milvus_ms")) for r in rows]
    seaweed = [_f(r.get("rag_seaweed_ms")) for r in rows]
    rag_retrieve = [_f(r.get("rag_retrieve_ms")) for r in rows]
    rag_format   = [_f(r.get("rag_format_ms"))   for r in rows]
    rag_chunks   = [_i(r.get("rag_num_chunks"))  for r in rows if r.get("rag_num_chunks")]
    rag_score    = [_f(r.get("rag_top_score"))   for r in rows if r.get("rag_top_score")]
    cembed = [_f(r.get("cache_embed_ms")) for r in rows]
    cmilvus = [_f(r.get("cache_milvus_ms")) for r in rows]
    cmongo = [_f(r.get("cache_mongo_ms")) for r in rows]
    cache_write  = [_f(r.get("cache_write_ms"))  for r in rows]
    sc_lookup = [_f(r.get("semantic_cache_lookup_ms")) for r in rows]
    q_tok = [_i(r.get("query_words"))            for r in rows if r.get("query_words")]
    orig_tok = [_i(r.get("original_prompt_tokens")) for r in rows if r.get("original_prompt_tokens")]
    aug_tok  = [_i(r.get("augmented_prompt_tokens")) for r in rows if r.get("augmented_prompt_tokens")]
    # Streaming-only fields — present only in --stream runs. Non-streaming rows have "".
    # Use truthiness check: empty string is falsy, so r.get("field") skips "" correctly.
    real_ttft    = [_f(r["ttft_ms"])               for r in rows if r.get("ttft_ms")]
    gen_ms_vals  = [_f(r["generation_ms"])          for r in rows if r.get("generation_ms")]
    ichunk_vals  = [_f(r["stream_inter_chunk_ms"])  for r in rows if r.get("stream_inter_chunk_ms")]
    real_tpot_v  = [_f(r["tpot_ms"])               for r in rows if r.get("tpot_ms")]
    nchunks_vals = [_i(r["n_chunks_with_content"])  for r in rows if r.get("n_chunks_with_content")]
    is_streaming = bool(real_ttft)
    routes = {}
    for r in rows:
        routes[r.get("route", "")] = routes.get(r.get("route", ""), 0) + 1
    cache_hits = sum(1 for r in rows if r.get("cache_hit") == "True")
    max_tok = _i(rows[0].get("max_tokens"))
    eos_count = sum(1 for n in nout if n and n < max_tok)
    # tpot requires BOTH backend_ms > 0 AND n_output_tokens > 0 (cache hits have n>0 but mb=0)
    _tpot_valid = [m / n for m, n in zip(mb, nout) if n > 0 and m > 0]
    return {
        "n": len(rows),
        "max_tokens": max_tok,
        "routes": routes,
        "cache_hit_pct": round(cache_hits / len(rows) * 100, 1),
        "bypass_rag":   rows[0].get("bypass_rag", ""),
        "bypass_cache": rows[0].get("bypass_cache", ""),
        "e2e_min":  round(min(e2e), 1)              if e2e else 0,
        "e2e_p50":  round(_pct(e2e, 0.50), 1),
        "e2e_p95":  round(_pct(e2e, 0.95), 1),
        "e2e_p99":  round(_pct(e2e, 0.99), 1),
        "e2e_max":  round(max(e2e), 1)              if e2e else 0,
        "e2e_mean": round(statistics.mean(e2e), 1) if e2e else 0,
        "mb_min":   round(min(mb), 1)               if mb  else 0,
        "mb_p50":   round(_pct(mb, 0.50), 1),
        "mb_max":   round(max(mb), 1)               if mb  else 0,
        "mb_mean":  round(statistics.mean(mb), 1)  if mb  else 0,
        "fe_min":   round(min(ttft), 1)             if ttft else 0,
        "fe_p50":   round(_pct(ttft, 0.50), 1),
        "fe_max":   round(max(ttft), 1)             if ttft else 0,
        "fe_mean":  round(statistics.mean(ttft), 1) if ttft else 0,
        # keep legacy keys for backward compat with any callers
        "ttft_p50": round(_pct(ttft, 0.50), 1),
        "ttft_mean": round(statistics.mean(ttft), 1) if ttft else 0,
        "n_out_mean": round(statistics.mean(nout), 1) if nout else 0,
        "n_out_max":  max(nout) if nout else 0,
        "eos_pct": round(eos_count / len(rows) * 100, 1) if rows else 0,
        "rag_embed_mean":    round(statistics.mean(embed), 2) if embed else 0,
        "rag_milvus_mean":   round(statistics.mean(milvus), 2) if milvus else 0,
        "rag_seaweed_mean":  round(statistics.mean(seaweed), 2) if seaweed else 0,
        "rag_retrieve_mean": round(statistics.mean(rag_retrieve), 2) if rag_retrieve else 0,
        "rag_format_mean":   round(statistics.mean(rag_format), 3)   if rag_format else 0,
        "rag_chunks_mean":   round(statistics.mean(rag_chunks), 2)   if rag_chunks else None,
        "rag_score_mean":    round(statistics.mean(rag_score), 4)    if rag_score else None,
        "cache_embed_mean":  round(statistics.mean(cembed), 2) if cembed else 0,
        "cache_milvus_mean": round(statistics.mean(cmilvus), 2) if cmilvus else 0,
        "cache_mongo_mean":  round(statistics.mean(cmongo), 2) if cmongo else 0,
        "cache_write_mean":  round(statistics.mean(cache_write), 3)  if cache_write else 0,
        "sc_lookup_mean":    round(statistics.mean(sc_lookup), 2) if sc_lookup else 0,
        "query_words_mean":     round(statistics.mean(q_tok), 1)    if q_tok else None,
        "orig_tokens_mean":      round(statistics.mean(orig_tok), 1) if orig_tok else None,
        "aug_tokens_mean":       round(statistics.mean(aug_tok), 1)  if aug_tok else None,
        # Latency percentiles for backend + ttft (we already have e2e ones)
        "ttft_p95":  round(_pct(ttft, 0.95), 1) if ttft else None,
        "ttft_p99":  round(_pct(ttft, 0.99), 1) if ttft else None,
        "mb_p95":    round(_pct(mb,   0.95), 1) if mb   else None,
        "mb_p99":    round(_pct(mb,   0.99), 1) if mb   else None,
        # Per-token latency — requires both backend_ms > 0 AND n_output_tokens > 0.
        # Cache hits: nout > 0 but mb = 0 (no backend call) → _tpot_valid is empty → None.
        # Streaming runs without usage: nout = 0 → None.
        "tpot_p50":  round(_pct(_tpot_valid, 0.50), 2) if _tpot_valid else None,
        "tpot_mean": round(statistics.mean(_tpot_valid), 2) if _tpot_valid else None,
        "n_out_available": bool(any(n > 0 for n in nout)),
        # Per-component p95/p99 (we already had only mean) — directly from CSV
        "rag_embed_p95":   round(_pct(embed, 0.95), 2)   if embed   else None,
        "rag_milvus_p95":  round(_pct(milvus, 0.95), 2)  if milvus  else None,
        "rag_seaweed_p95": round(_pct(seaweed, 0.95), 2) if seaweed else None,
        "cache_embed_p95":  round(_pct(cembed, 0.95), 2)  if cembed  else None,
        "cache_milvus_p95": round(_pct(cmilvus, 0.95), 2) if cmilvus else None,
        "cache_mongo_p95":  round(_pct(cmongo, 0.95), 2)  if cmongo  else None,
        # Prompt inflation: how much RAG context inflates the prompt
        "prompt_inflation": round(statistics.mean(aug_tok) / statistics.mean(orig_tok), 2)
                            if aug_tok and orig_tok and statistics.mean(orig_tok) > 0 else None,
        "is_streaming": is_streaming,
        "real_ttft_min":  round(min(real_ttft), 1)              if real_ttft else None,
        "real_ttft_p50":  round(_pct(real_ttft, 0.50), 1)       if real_ttft else None,
        "real_ttft_max":  round(max(real_ttft), 1)              if real_ttft else None,
        "real_ttft_mean": round(statistics.mean(real_ttft), 1)  if real_ttft else None,
        "gen_ms_p50":     round(_pct(gen_ms_vals, 0.50), 1)     if gen_ms_vals else None,
        "gen_ms_mean":    round(statistics.mean(gen_ms_vals), 1) if gen_ms_vals else None,
        "ichunk_p50":     round(_pct(ichunk_vals, 0.50), 2)     if ichunk_vals else None,
        "real_tpot_p50":  round(_pct(real_tpot_v, 0.50), 2)    if real_tpot_v else None,
        "real_tpot_mean": round(statistics.mean(real_tpot_v), 2) if real_tpot_v else None,
        "n_chunks_mean":  round(statistics.mean(nchunks_vals), 1) if nchunks_vals else None,
        # Throughput for the cell as a whole (uses dram_wall_sec via load_cell post-merge)
        "n_rows": len(rows),
    }


def latency_stats_isolated(rows: List[Dict], mode: str) -> Dict[str, Any]:
    """Stats from bge_isolated (text_words, embed_ms) or hnsw_isolated (text_words, hnsw_ms, num_hits)."""
    if not rows:
        return {}
    if mode.startswith("bge"):
        ms_field = "embed_ms"
    else:
        ms_field = "hnsw_ms"
    vals = [_f(r.get(ms_field, 0)) for r in rows]
    small_n = len(vals) < 20
    return {
        "n":      len(rows),
        "small_n": small_n,
        "median": round(_pct(vals, 0.50), 2),
        # p95/p99 only meaningful for n≥20; for small n, equals max
        "p95":    round(_pct(vals, 0.95), 2) if not small_n else None,
        "p99":    round(_pct(vals, 0.99), 2) if not small_n else None,
        "mean":   round(statistics.mean(vals), 2) if vals else 0,
        "max":    round(max(vals), 2) if vals else 0,
        "min":    round(min(vals), 2) if vals else 0,
    }


# ── cell discovery ────────────────────────────────────────────────────────────

ISOLATED_PREFIXES = ("bge_isolated", "hnsw_isolated")

def audit_cell(cell: Dict[str, Any], csv_rows: List[Dict], expected_count: int) -> List[Dict[str, str]]:
    """Run sanity checks on a loaded cell. Returns list of {severity, msg, remediation} findings.
    severity: "error" (data invalid), "warn" (suspicious but possibly legitimate), "info" (notable but OK)."""
    findings: List[Dict[str, str]] = []
    name = cell["name"]
    mode = cell["mode"]
    is_iso = cell["is_isolated"]
    hw = cell["hw"]
    s = cell["stats"]

    # ── A. Expected perf-pass count ──
    # 4 passes expected: pass1 (IPC), pass2a+2b (cache/stalls), pass3 (DRAM), pass4 (MLP).
    # SIMD pass permanently dropped — p3 is always {} for new runs.
    expected_passes = 1 if is_iso else 4
    min_acceptable  = 1 if is_iso else 3
    if cell["perf_passes"] < min_acceptable:
        findings.append({"severity": "error",
            "msg": f"only {cell['perf_passes']}/{expected_passes} perf passes ran (cell incomplete)",
            "remediation": f"wait for run_benchmark.sh to finish, OR run "
                           f"<code>scripts/run_benchmark.sh {cell['mode']}_{cell['bucket']}</code> "
                           f"to retry just this cell"})
    elif cell["perf_passes"] < expected_passes:
        findings.append({"severity": "warn",
            "msg": f"{cell['perf_passes']}/{expected_passes} perf passes — likely pass3 (DRAM) or pass4 (MLP) absent. "
                   f"Other metrics unaffected."})

    # ── B. CSV row count ──
    if csv_rows and expected_count and len(csv_rows) != expected_count:
        if len(csv_rows) < expected_count:
            findings.append({"severity": "warn",
                "msg": f"CSV has {len(csv_rows)}/{expected_count} rows (short)"})
        else:
            findings.append({"severity": "info",
                "msg": f"CSV has {len(csv_rows)}/{expected_count} rows (extra)"})

    if not csv_rows:
        return findings  # nothing further to check

    # ── C/D. HTTP status + error column ──
    bad_status = sum(1 for r in csv_rows if r.get("http_status") and r["http_status"] != "200")
    if bad_status:
        findings.append({"severity": "error",
            "msg": f"{bad_status}/{len(csv_rows)} rows had non-200 HTTP status"})
    err_count = sum(1 for r in csv_rows if r.get("error", "").strip())
    if err_count:
        findings.append({"severity": "error",
            "msg": f"{err_count}/{len(csv_rows)} rows reported errors"})

    if is_iso:
        return findings  # isolated cells have a different schema; routes don't apply

    # ── E. Route ↔ mode ──
    # Routes that are legitimate given each mode's bypass flags:
    #   sc_a (bypass_rag=True,  bypass_cache=False): cache hit = semantic_cache;
    #                                                cache miss = plain_backend (RAG bypassed).
    #   sc_b (bypass_rag=False, bypass_cache=False): cache hit = semantic_cache;
    #                                                cache miss with RAG hit = rag_plus_backend;
    #                                                cache miss with no RAG (below threshold) = plain_backend.
    routes = {}
    for r in csv_rows:
        rt = r.get("route", "")
        routes[rt] = routes.get(rt, 0) + 1
    n = len(csv_rows)
    expected_routes = {
        "rag":         {"rag_plus_backend"},
        "sc_a":        {"semantic_cache", "plain_backend"},
        "sc_b":        {"semantic_cache", "rag_plus_backend", "plain_backend"},
        "llm_direct":  {"plain_backend"},
    }
    if mode in expected_routes:
        unexpected = set(routes) - expected_routes[mode]
        if unexpected:
            findings.append({"severity": "error",
                "msg": f"unexpected routes for {mode}: {unexpected} (got {routes})"})

    # SC-specific: report cache hit rate as info — low hit rate is real workload
    # behavior (paraphrase corpus + BGE-base + similarity threshold), not a bug.
    if mode in ("sc_a", "sc_b"):
        hits = sum(1 for r in csv_rows if str(r.get("cache_hit", "")).lower() == "true")
        pct = round(hits / n * 100, 1) if n else 0
        findings.append({"severity": "info",
            "msg": f"{mode} cache hit rate: {hits}/{n} ({pct}%). "
                   f"Routes: {dict(sorted(routes.items()))}. "
                   f"Misses are routed per bypass flags (sc_a → plain_backend, sc_b → rag_plus_backend or plain_backend)."})

    # ── F/G. bypass flags ──
    bypass_rag   = {r.get("bypass_rag")   for r in csv_rows}
    bypass_cache = {r.get("bypass_cache") for r in csv_rows}
    expected_bypass = {
        "rag":        ("False", "True"),
        "sc_a":       ("True",  "False"),
        "sc_b":       ("False", "False"),
        "llm_direct": ("True",  "True"),
    }
    if mode in expected_bypass:
        e_rag, e_cache = expected_bypass[mode]
        if bypass_rag != {e_rag}:
            findings.append({"severity": "error",
                "msg": f"bypass_rag flag mismatch for {mode}: expected {e_rag}, got {bypass_rag}"})
        if bypass_cache != {e_cache}:
            findings.append({"severity": "error",
                "msg": f"bypass_cache flag mismatch for {mode}: expected {e_cache}, got {bypass_cache}"})

    # ── H/I. Per-component telemetry ↔ mode ──
    def _nonzero(field: str) -> int:
        return sum(1 for r in csv_rows if _f(r.get(field)) > 0)
    rag_embed_nz = _nonzero("rag_embed_ms")
    cache_embed_nz = _nonzero("cache_embed_ms")
    expected_telemetry = {
        # mode → (rag_embed should be nonzero?, cache_embed should be nonzero?)
        "rag":        (True,  False),
        "sc_a":       (False, True),
        "sc_b":       (True,  True),    # production flow runs RAG first
        "llm_direct": (False, False),
    }
    if mode in expected_telemetry:
        rag_should, cache_should = expected_telemetry[mode]
        if rag_should and rag_embed_nz < n * 0.95:
            findings.append({"severity": "error",
                "msg": f"rag_embed_ms expected nonzero for {mode} but only {rag_embed_nz}/{n} are"})
        if not rag_should and rag_embed_nz > n * 0.05:
            findings.append({"severity": "error",
                "msg": f"rag_embed_ms expected zero for {mode} but {rag_embed_nz}/{n} are nonzero"})
        if cache_should and cache_embed_nz < n * 0.95:
            findings.append({"severity": "error",
                "msg": f"cache_embed_ms expected nonzero for {mode} but only {cache_embed_nz}/{n} are"})
        if not cache_should and cache_embed_nz > n * 0.05:
            findings.append({"severity": "error",
                "msg": f"cache_embed_ms expected zero for {mode} but {cache_embed_nz}/{n} are nonzero"})

    # ── K. Latency monotonicity ──
    e2e_vals = sorted(_f(r.get("e2e_ms")) for r in csv_rows if r.get("e2e_ms"))
    if e2e_vals:
        p50 = e2e_vals[len(e2e_vals)//2]
        p95 = e2e_vals[int(len(e2e_vals)*0.95)]
        p99 = e2e_vals[int(len(e2e_vals)*0.99)]
        if not (p50 <= p95 <= p99):
            findings.append({"severity": "error",
                "msg": f"e2e percentile order violated: p50={p50:.1f} p95={p95:.1f} p99={p99:.1f}"})

    # ── L/M. e2e contains its sub-phases ──
    over_count = sum(1 for r in csv_rows
                     if _f(r.get("model_backend_http_ms")) > _f(r.get("e2e_ms")) + 1.0)
    if over_count:
        findings.append({"severity": "error",
            "msg": f"{over_count} rows where model_backend_http_ms > e2e_ms (impossible)"})

    # ── N/O. DRAM peak ≥ avg ──
    if hw.get("dram_peak_read_gbs") is not None and hw.get("dram_avg_read_gbs") is not None:
        if hw["dram_peak_read_gbs"] < hw["dram_avg_read_gbs"]:
            findings.append({"severity": "error",
                "msg": f"DRAM peak_read ({hw['dram_peak_read_gbs']}) < avg_read ({hw['dram_avg_read_gbs']})"})
    if hw.get("dram_peak_write_gbs") is not None and hw.get("dram_avg_write_gbs") is not None:
        if hw["dram_peak_write_gbs"] < hw["dram_avg_write_gbs"]:
            findings.append({"severity": "error",
                "msg": f"DRAM peak_write ({hw['dram_peak_write_gbs']}) < avg_write ({hw['dram_avg_write_gbs']})"})

    # ── P. Memory pyramid: L1 ≥ L2 ≥ L3 ──
    l1, l2, l3 = hw.get("l1_hits"), hw.get("l2_hits"), hw.get("l3_hits")
    if l1 is not None and l2 is not None and l3 is not None:
        if not (l1 >= l2 >= l3):
            findings.append({"severity": "warn",
                "msg": f"memory pyramid violation: L1={fmt_int(l1)}, L2={fmt_int(l2)}, L3={fmt_int(l3)}"})

    # ── Q. IPC plausibility ──
    if hw.get("ipc") is not None:
        if hw["ipc"] < 0.05 or hw["ipc"] > 5.0:
            findings.append({"severity": "warn",
                "msg": f"IPC {hw['ipc']} outside plausible range (0.05–5.0)"})

    # ── R. AVX% range ──
    for k in ("avx512_pct", "avx256_pct", "amx_pct"):
        v = hw.get(k)
        if v is not None and (v < 0 or v > 100):
            findings.append({"severity": "warn",
                "msg": f"{k}={v} outside [0, 100]"})

    # ── S/T. Stall ratios in [0, 100] ──
    # When pct > 100%, this is normally impossible — but here it usually means
    # the metric's numerator (e.g. stalls_total from pass2) was measured in a
    # different perf invocation than its denominator (cycles from pass1). The
    # two passes have different wall windows AND cover different request mixes,
    # so the cross-pass ratio can exceed 100%. We flag these so the underlying
    # measurement quirk is visible rather than silently producing misleading
    # percentages in the table.
    for k in ("mem_stall_pct", "store_stall_pct", "ports_util_pct",
              "stalls_tot_pct", "stalls_l3_pct", "spec_waste_pct"):
        v = hw.get(k)
        if v is not None and (v < 0 or v > 100):
            findings.append({"severity": "warn",
                "msg": f"{k}={v}% outside [0, 100] — likely cross-pass cycle-count mismatch "
                       f"(numerator from pass2/3/5, denominator from pass1; see report caveat)",
                "remediation": f"add <code>cycles</code> as an event to pass2/3/5 in "
                               f"<code>scripts/run_benchmark.sh</code> so each pass has its own "
                               f"in-pass denominator. Then re-run the affected cell: "
                               f"<code>scripts/run_benchmark.sh {cell['mode']}_{cell['bucket']}</code>"})

    # ── Z. Cross-pass cycles divergence (info) — a soft predictor of where
    #       any percentage built from "pass-X numerator / pass1 cycles" could be biased.
    # Detect when: pass1 had a dominant idle composition vs pass2's. We approximate
    # this by checking if stalls_tot_pct differs from mem_stall_pct by an unusual
    # amount (both should track each other in a real measurement).
    sp = hw.get("stalls_tot_pct"); mp = hw.get("mem_stall_pct")
    if sp is not None and mp is not None and sp <= 100 and mp <= 100 and abs(sp - mp) > 50:
        findings.append({"severity": "info",
            "msg": f"stalls_tot_pct ({sp}%) and mem_stall_pct ({mp}%) differ by >50 points — "
                   f"either a real workload phase change between pass2 and pass4, or cross-pass "
                   f"cycle drift. Treat both as approximate.",
            "remediation": "add <code>cycles</code> as an event in every pass for exact within-pass percentages"})

    # ── X. n_output_tokens never exceeds max_tokens ──
    bad_n = sum(1 for r in csv_rows
                if _i(r.get("n_output_tokens", 0)) > _i(r.get("max_tokens", 0)) and _i(r.get("max_tokens", 0)) > 0)
    if bad_n:
        findings.append({"severity": "error",
            "msg": f"{bad_n} rows where n_output_tokens > max_tokens (impossible)"})

    # ── Y. wall_sec for pass3 (sanity: should be > 0) ──
    if hw.get("dram_wall_sec") is not None and hw["dram_wall_sec"] <= 0:
        findings.append({"severity": "error",
            "msg": f"pass3 wall_sec={hw['dram_wall_sec']} (no measurement window)"})

    return findings


def cross_cell_audit(cells_by_tier: Dict[str, List[Dict]], stream: Dict, tma: Dict = None) -> List[Dict[str, str]]:
    """Cross-cell consistency checks. Returns list of findings keyed by 'cross-cell' scope."""
    findings: List[Dict[str, str]] = []

    # ── TMA file integrity: any tier/mode whose slots or toplev parse returned _failed
    if tma:
        for tier, tier_data in tma.items():
            for mode, fields in tier_data.items():
                if fields.get("_failed"):
                    findings.append({"severity": "warn",
                        "msg": f"TMA capture for {tier}/{mode}/medium failed: {fields['_failed']}",
                        "remediation": f"re-run just this TMA cell with the same command in <code>scripts/run_tma_extra.sh</code>"})

    # ── TPOT consistency across complete LLM-active cells ──
    tpots = []
    for tier, cells in cells_by_tier.items():
        for c in cells:
            if c["is_isolated"] or not c["is_complete"] or not c["stats"]: continue
            tpot = c["stats"].get("tpot_p50")
            if tpot and tpot > 0:
                tpots.append((f'{tier}/{c["name"]}', tpot))
    if len(tpots) >= 3:
        values = [t for _, t in tpots]
        mean = statistics.mean(values)
        max_dev = max(abs(t - mean) for t in values)
        # GPU rate should be very stable; >20% deviation = suspicious
        if max_dev / mean > 0.20:
            outliers = [name for name, t in tpots if abs(t - mean) / mean > 0.20]
            findings.append({"severity": "warn",
                "msg": f"TPOT p50 varies >20% across cells (mean {mean:.1f} ms/tok). Outliers: {outliers}"})

    # ── DRAM avg/peak vs STREAM Triad ──
    triad = stream.get("triad_gbs") if stream else None
    if triad:
        for tier, cells in cells_by_tier.items():
            for c in cells:
                if c["is_isolated"]: continue
                avg = c["hw"].get("dram_avg_total_gbs")
                peak = (c["hw"].get("dram_peak_read_gbs") or 0) + (c["hw"].get("dram_peak_write_gbs") or 0)
                if avg is not None and avg > triad:
                    findings.append({"severity": "error",
                        "msg": f"{tier}/{c['name']}: DRAM avg total ({avg} GB/s) exceeds STREAM Triad ({triad} GB/s) — impossible"})
                if peak > triad * 1.5:  # peak above ceiling somewhat OK due to short-burst cas_count, but >1.5× is suspect
                    findings.append({"severity": "warn",
                        "msg": f"{tier}/{c['name']}: DRAM peak total ({peak:.1f} GB/s) exceeds STREAM Triad × 1.5 ({triad*1.5:.1f}) — verify"})

    # ── Per-component RAG latencies should be similar across same-bucket cells of the same tier ──
    # For each tier, group rag cells and compare rag_milvus_ms (which should be input-independent)
    for tier, cells in cells_by_tier.items():
        rag_cells = [c for c in cells if c["mode"] == "rag" and c["is_complete"] and c["stats"]]
        if len(rag_cells) >= 2:
            milvus_means = [c["stats"].get("rag_milvus_mean", 0) for c in rag_cells]
            if max(milvus_means) - min(milvus_means) > 1.0:  # >1ms range across buckets
                findings.append({"severity": "info",
                    "msg": f"{tier}: rag_milvus_mean varies > 1ms across buckets ({[round(v,2) for v in milvus_means]}) — Milvus should be input-independent"})

    return findings


# milvus_etcd, milvus_minio, seaweed_master dropped — idle control-plane services
# are no longer deep-PMU measured (their counter data was noise).
ALL_POD_KEYS = ("fastapi", "milvus", "mongodb",
                "seaweed_volume", "seaweed_filer", "llmd_gateway", "vllm")
# vLLM runs on a Nitro/GPU node — no hardware PMU, software events only.
_SW_ONLY_PODS = {"vllm"}


def load_cell(cell_dir: Path) -> Dict[str, Any]:
    name = cell_dir.name.replace("cell_", "")
    is_isolated = any(name.startswith(p) for p in ISOLATED_PREFIXES)

    def _merge_passes(a: Optional[Dict], b: Optional[Dict]) -> Optional[Dict]:
        """Merge two split perf pass dicts into one. Returns None if both empty."""
        merged = {**(a or {}), **(b or {})}
        return merged if merged else None

    # pass4 raw event names → normalized names expected by derive_hw
    _P5_KEY_MAP = {
        "mem_load_retired.l1_hit":  "l1_hit",
        "mem_load_retired.l2_hit":  "l2_hit",
        "mem_load_retired.l3_hit":  "l3_hit",
        "exe_activity.bound_on_loads":  "bound_on_loads",
        "exe_activity.bound_on_stores": "bound_on_stores",
        "exe_activity.1_ports_util":    "ports_util",
        "l1d_pend_miss.pending":        "l1d_pend",
        "l1d_pend_miss.pending_cycles": "l1d_pend_cycles",
    }

    def _normalize_p5(d: Optional[Dict]) -> Optional[Dict]:
        """Rename raw pass4 event keys to the normalized names derive_hw expects.
        Also adds l1_pct/l2_pct/l3_pct derived from load counts."""
        if not d:
            return d
        out = {_P5_KEY_MAP.get(k, k): v for k, v in d.items()}
        l1 = out.get("l1_hit", 0) or 0
        l2 = out.get("l2_hit", 0) or 0
        l3 = out.get("l3_hit", 0) or 0
        total = l1 + l2 + l3
        if total:
            out["l1_pct"] = round(l1 / total * 100, 2)
            out["l2_pct"] = round(l2 / total * 100, 2)
            out["l3_pct"] = round(l3 / total * 100, 2)
        return out

    # ── Primary (backward compat): load fastapi files without pod suffix ──
    # pass2a+pass2b replace pass2; pass4a+pass4b are the old intermediate split
    # format (before 5a/5b were merged into a single pass4). Falls back to
    # legacy pass2 files for old runs.
    _p2a = parse_perf_totals(cell_dir / "perf_pass2a.txt")
    _p2b = parse_perf_totals(cell_dir / "perf_pass2b.txt")
    _p4a = parse_perf_totals(cell_dir / "perf_pass4a.txt")
    _p4b = parse_perf_totals(cell_dir / "perf_pass4b.txt")
    _p4_merged = _merge_passes(_p4a, _p4b)
    perf = {
        "p1": parse_perf_totals(cell_dir / "perf_pass1.txt"),
        "p2": _merge_passes(_p2a, _p2b) or parse_perf_totals(cell_dir / "perf_pass2.txt"),
        "p3": {},  # SIMD pass permanently dropped — always empty for new runs
        "p4": parse_pass3_imc(cell_dir / "perf_pass3.txt"),
        "p5": _normalize_p5(_p4_merged) if _p4_merged else parse_pass4_mem(cell_dir / "perf_pass4.txt"),
    }
    hw = derive_hw(perf["p1"], perf["p2"], perf["p3"], perf["p4"], perf["p5"])

    # ── Per-pod perf files (new multi-pod format) ──────────────────────────
    # Files: perf_pass1_fastapi.txt, perf_pass2a_fastapi.txt, etc.
    # Pass 3 (DRAM) node-wide: perf_pass3_node.txt (no per-pod suffix)
    per_pod_hw: Dict[str, Any] = {}
    p4_node = parse_pass3_imc(cell_dir / "perf_pass3_node.txt")
    for pod_key in ALL_POD_KEYS:
        p1_pod = parse_perf_totals(cell_dir / f"perf_pass1_{pod_key}.txt")
        if not p1_pod:
            continue
        _p2a_pod = parse_perf_totals(cell_dir / f"perf_pass2a_{pod_key}.txt")
        _p2b_pod = parse_perf_totals(cell_dir / f"perf_pass2b_{pod_key}.txt")
        p2_pod = _merge_passes(_p2a_pod, _p2b_pod) or parse_perf_totals(cell_dir / f"perf_pass2_{pod_key}.txt")
        p3_pod = {}  # SIMD pass dropped — no per-pod pass3 files exist for new runs
        _p4a_pod = parse_perf_totals(cell_dir / f"perf_pass4a_{pod_key}.txt")
        _p4b_pod = parse_perf_totals(cell_dir / f"perf_pass4b_{pod_key}.txt")
        _p4_pod_merged = _merge_passes(_p4a_pod, _p4b_pod)
        p5_pod = _normalize_p5(_p4_pod_merged) if _p4_pod_merged else parse_pass4_mem(cell_dir / f"perf_pass4_{pod_key}.txt")
        per_pod_hw[pod_key] = derive_hw(p1_pod, p2_pod, p3_pod, p4_node, p5_pod)

    # If fastapi pod file exists, use it as the primary hw (overrides compat file)
    if "fastapi" in per_pod_hw:
        hw = per_pod_hw["fastapi"]

    # Determine completeness: a cell is complete for latency reporting if CSV data exists.
    # perf_complete tracks whether hardware counter data is usable (≥4/5 passes).
    # Pass3 (SIMD) may be absent on kernels where bf16/amx events are unsupported.
    expected_passes = 1 if is_isolated else 5
    perf_passes = sum(1 for p in perf.values() if p)
    perf_complete = perf_passes >= (1 if is_isolated else 4)
    # is_complete = True as long as we have CSV rows — latency is always shown.
    is_complete = True  # refined below once csvs are loaded

    # CSV source: ONLY use pass4/ (or root for isolated). Never silently fall back
    # to pass1's CSV for hw cells — that mixes the latency from a query phase
    # measured WITH pass1's perf events vs a different one with pass4's.
    csvs: List[Path] = []
    if is_isolated:
        csvs = sorted(p for p in cell_dir.glob("*.csv") if p.name != "vllm_metrics.csv")
    else:
        # Use root (pass1) CSV for latency — most representative, least warmed-up state.
        # Subsequent passes see progressive warmup (caches hot, JIT compiled) so their
        # latency is lower and not representative of deployment. Perf counters are read
        # from their own pass files independently of which CSV we use for latency.
        csvs = sorted(p for p in cell_dir.glob("*.csv") if p.name != "vllm_metrics.csv")
        if not csvs:
            for subdir in ("pass4b", "pass4a", "pass4"):
                candidate = sorted((cell_dir / subdir).glob("*.csv")) if (cell_dir / subdir).exists() else []
                if candidate:
                    csvs = candidate
                    break
    csv_rows: List[Dict] = []
    if csvs:
        try:
            csv_rows = list(csv.DictReader(open(csvs[-1])))
        except Exception:
            csv_rows = []

    if is_isolated:
        if name.startswith("bge"):
            stats = latency_stats_isolated(csv_rows, "bge")
        else:
            stats = latency_stats_isolated(csv_rows, "hnsw")
    else:
        stats = latency_stats_main(csv_rows)

    # A cell is complete for latency reporting if CSV rows exist.
    # perf_complete tracks hw-counter completeness separately.
    is_complete = bool(csv_rows)

    # mode and bucket
    parts = name.split("_")
    if name.startswith(("bge_isolated", "hnsw_isolated")):
        mode = "_".join(parts[:2])
        bucket = "_".join(parts[2:])
    elif name.startswith(("sc_a", "sc_b", "llm_direct")):
        mode = "_".join(parts[:2])
        bucket = "_".join(parts[2:])
    else:
        mode = parts[0]
        bucket = "_".join(parts[1:])

    gpu_dmon = parse_gpu_dmon(cell_dir / "gpu_dmon.txt")
    gpu_pmon = parse_gpu_pmon(cell_dir / "gpu_pmon.txt")
    vllm_metrics = parse_vllm_metrics(cell_dir / "vllm_metrics_summary.json")
    vllm_interval = parse_vllm_interval(cell_dir / "perf_pass1_vllm_interval.txt")

    cell = {
        "name": name,
        "mode": mode,
        "bucket": bucket,
        "is_isolated": is_isolated,
        "perf_passes": perf_passes,
        "perf_complete": perf_complete,
        "is_complete": is_complete,
        "stats": stats,
        "hw": hw,
        "per_pod_hw": per_pod_hw,
        "csv_rows": len(csv_rows),
        "raw_rows": csv_rows,
        "gpu_dmon": gpu_dmon,
        "gpu_pmon": gpu_pmon,
        "vllm": vllm_metrics,
        "vllm_interval": vllm_interval,
    }
    # Expected count: use actual CSV row count as baseline (no hardcoded floor).
    # The check in audit_cell only fires if expected_count > 0 and actual != expected,
    # so passing 0 skips the count check entirely — correct when n is intentionally small.
    cell["audit"] = audit_cell(cell, csv_rows, 0)
    return cell


def load_run(run_dir: Path, tiers: Optional[List[str]] = None) -> Dict[str, Any]:
    all_tiers = tiers or ["tok64", "tok192", "tok320"]
    cells = {}  # tier → list of cells
    for tier in all_tiers:
        tdir = run_dir / tier
        if not tdir.exists():
            continue
        cells[tier] = []
        for cd in sorted(tdir.glob("cell_*")):
            try:
                cells[tier].append(load_cell(cd))
            except Exception as e:
                print(f"WARN: failed to load {cd}: {e}", file=sys.stderr)

    # TMA — base files (tok64) and tier-suffixed extras (tok192 / tok320)
    # Also load per-pod TMA files: tma_toplev_{mode}_{pod_key}.txt
    tma_dir = run_dir / "tma"
    tma = {}              # tier "tok64" → {mode → {fields}}
    # per_pod_tma: tier → mode → pod_key → {fields}
    per_pod_tma: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if tma_dir.exists():
        # Base (tok64): fastapi backward-compat files (no pod suffix)
        tma["tok64"] = {}
        per_pod_tma["tok64"] = {}
        for path_mode in ("rag", "llm_direct", "sc_a"):
            slots = parse_tma_slots(tma_dir / f"tma_slots_{path_mode}.txt")
            toplev = parse_tma_toplev(tma_dir / f"tma_toplev_{path_mode}.txt")
            if slots or toplev:
                tma["tok64"][path_mode] = {**slots,
                    **{f"toplev_{k}": v for k, v in toplev.items()}}
            # Per-pod TMA files
            per_pod_tma["tok64"].setdefault(path_mode, {})
            for pod_key in ALL_POD_KEYS:
                pod_slots = parse_tma_slots(tma_dir / f"tma_slots_{path_mode}_{pod_key}.txt")
                pod_toplev = parse_tma_toplev(tma_dir / f"tma_toplev_{path_mode}_{pod_key}.txt")
                if pod_slots or pod_toplev:
                    per_pod_tma["tok64"][path_mode][pod_key] = {
                        **pod_slots, **{f"toplev_{k}": v for k, v in pod_toplev.items()}
                    }
        # Tier-suffixed (tok192/tok320) — written by the integrated run_tma() per tier.
        for tier in ("tok192", "tok320"):
            tier_data = {}
            for path_mode in ("rag", "llm_direct", "sc_a"):
                slots = parse_tma_slots(tma_dir / f"tma_slots_{path_mode}_{tier}.txt")
                toplev = parse_tma_toplev(tma_dir / f"tma_toplev_{path_mode}_{tier}.txt")
                if slots or toplev:
                    tier_data[path_mode] = {**slots,
                        **{f"toplev_{k}": v for k, v in toplev.items()}}
            if tier_data:
                tma[tier] = tier_data

    # run_info
    info = {}
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            pass

    # STREAM calibration (real measured ceiling on this host)
    stream = parse_stream(run_dir / "calibration" / "stream.txt")

    return {"run_dir": str(run_dir), "cells": cells, "tma": tma,
            "per_pod_tma": per_pod_tma, "info": info, "stream": stream}


# ── HTML generation ───────────────────────────────────────────────────────────

CSS = """
:root {
  --bg:#0d1117; --panel:#161b22; --panel-2:#1c232c; --border:#30363d;
  --text:#e6edf3; --text-dim:#8b949e; --accent:#58a6ff;
  --good:#3fb950; --warn:#d29922; --bad:#f85149;
  --rag:#58a6ff; --sca:#bc8cff; --scb:#ffa657; --llm:#f78166;
  --bge:#3fb950; --hnsw:#d29922;
}
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  font-size:14px; line-height:1.55; }
.container { max-width:1800px; margin:0 auto; padding:24px 32px 60px; }
h1,h2,h3 { font-weight:600; letter-spacing:-0.02em; }
h1 { font-size:28px; margin:0 0 6px; }
h2 { font-size:20px; margin:44px 0 16px; padding-bottom:8px; border-bottom:1px solid var(--border); }
h3 { font-size:16px; margin:24px 0 10px; color:var(--accent); }
p  { color:var(--text-dim); margin:6px 0; }
code { background:var(--panel-2); border:1px solid var(--border); border-radius:4px; padding:1px 5px; font-size:12px; }
.meta-card { background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:18px 22px; margin-bottom:12px;
  display:grid; grid-template-columns:repeat(4,1fr); gap:18px 24px; }
.meta-item .k { color:var(--text-dim); font-size:12px; text-transform:uppercase; letter-spacing:0.06em; }
.meta-item .v { color:var(--text); font-size:15px; font-weight:500; margin-top:2px; }
.note { background:var(--panel); border-left:3px solid var(--warn); border-radius:4px;
  padding:10px 14px; margin:10px 0; font-size:13px; color:var(--text-dim); }
table { width:100%; border-collapse:collapse; background:var(--panel);
  border:1px solid var(--border); border-radius:8px; overflow:hidden;
  font-size:12.5px; margin:10px 0 8px; }
th, td { padding:8px 11px; text-align:right; border-bottom:1px solid var(--border); }
th { background:var(--panel-2); color:var(--text-dim); font-weight:600; text-transform:uppercase;
  font-size:11px; letter-spacing:0.05em; }
th:first-child, td:first-child { text-align:left; }
tr:last-child td { border-bottom:none; }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
  font-weight:600; margin-right:6px; }
.tag-rag { background:rgba(88,166,255,0.18); color:var(--rag); }
.tag-sca { background:rgba(188,140,255,0.18); color:var(--sca); }
.tag-scb { background:rgba(255,166,87,0.18); color:var(--scb); }
.tag-llm { background:rgba(247,129,102,0.18); color:var(--llm); }
.tag-bge { background:rgba(63,185,80,0.18); color:var(--bge); }
.tag-hnsw { background:rgba(210,153,34,0.18); color:var(--hnsw); }
.tag-tier { background:var(--panel-2); color:var(--text-dim); border:1px solid var(--border); }
.chart-card { background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:12px 16px; margin:10px 0; }
/* Multi-column chart grids — gap replaces chart-card margin inside a grid */
.chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:12px; margin:10px 0; }
.chart-grid > .chart-card { margin:0; }
.chart-grid-2 { display:grid; grid-template-columns:1fr 2fr; gap:12px; margin:10px 0; }
.chart-grid-2 > .chart-card { margin:0; }
.grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
/* Collapsible full sections — h3 headers that fold away */
details.section-fold { border:1px solid var(--border); border-radius:8px; margin:8px 0; }
details.section-fold > summary {
  cursor:pointer; padding:10px 16px; font-size:13px; font-weight:600;
  color:var(--text); background:var(--panel-2); border-radius:8px; list-style:none;
  user-select:none; }
details.section-fold > summary::-webkit-details-marker { display:none; }
details.section-fold > summary::before { content:"▸ "; font-size:10px; color:var(--text-dim); }
details.section-fold[open] > summary { border-radius:8px 8px 0 0; border-bottom:1px solid var(--border); }
details.section-fold[open] > summary::before { content:"▾ "; }
details.section-fold > summary:hover { color:var(--accent); }
details.section-fold .fold-body { padding:12px 16px; }
.muted { color:var(--text-dim); font-size:12px; }
.kbd { background:var(--panel-2); border:1px solid var(--border); border-radius:3px;
  padding:0 5px; font-family:ui-monospace,monospace; font-size:11px; }
.flag { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px;
  font-weight:600; margin-left:4px; vertical-align:middle; }
.flag-error { background:rgba(248,81,73,0.18); color:var(--bad); border:1px solid rgba(248,81,73,0.3); }
.flag-warn  { background:rgba(210,153,34,0.18); color:var(--warn); border:1px solid rgba(210,153,34,0.3); }
.flag-info  { background:rgba(88,166,255,0.18); color:var(--accent); border:1px solid rgba(88,166,255,0.3); }
.audit-table td:nth-child(3) { text-align:left; }
.audit-error { color: var(--bad); }
.audit-warn  { color: var(--warn); }
.audit-info  { color: var(--accent); }
/* Collapsible detail blocks — dense numeric tables hide behind a click so the
   default view is charts + headlines. */
details.detail { background:transparent; border:none; margin:6px 0 14px; }
details.detail > summary {
  cursor:pointer; display:inline-block; padding:5px 12px; font-size:12px; font-weight:600;
  color:var(--text-dim); background:var(--panel-2); border:1px solid var(--border);
  border-radius:6px; list-style:none; user-select:none; }
details.detail > summary::-webkit-details-marker { display:none; }
details.detail > summary::before { content:"▸ "; }
details.detail[open] > summary::before { content:"▾ "; }
details.detail > summary:hover { color:var(--text); border-color:var(--accent); }
details.detail .detail-body { margin-top:6px; }
"""

MODE_TAG = {
    "rag":           "tag-rag",
    "sc_a":          "tag-sca",
    "sc_b":          "tag-scb",
    "llm_direct":    "tag-llm",
    "bge_isolated":  "tag-bge",
    "hnsw_isolated": "tag-hnsw",
}
BUCKET_ORDER = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
MODE_ORDER = {"rag": 0, "sc_a": 1, "sc_b": 2, "llm_direct": 3,
              "bge_isolated": 4, "hnsw_isolated": 5}

def collapsible(summary: str, inner: str, open_default: bool = False) -> str:
    """Hide a dense numeric table behind a native <details> toggle so the default
    report view is charts + headlines. Returns '' when there's nothing to show."""
    if not inner or not inner.strip():
        return ""
    op = " open" if open_default else ""
    return (f'<details class="detail"{op}><summary>{summary}</summary>'
            f'<div class="detail-body">{inner}</div></details>')


def section_fold(heading: str, inner_html: str) -> str:
    """Wrap an entire h3 section in a <details> block so it's collapsed by default.
    The summary line replaces the h3 heading. Returns inner_html unchanged if empty."""
    if not inner_html or not inner_html.strip():
        return ""
    return (f'<details class="section-fold"><summary>{heading}</summary>'
            f'<div class="fold-body">{inner_html}</div></details>')


def fmt_int(n):
    if n is None: return "—"
    n = int(n)
    if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
    if n >= 1_000_000:     return f"{n/1e6:.2f}M"
    if n >= 1_000:         return f"{n/1e3:.1f}K"
    return str(n)


def fmt_v(x, suffix=""):
    """Render a numeric metric for display. None → '—' (genuinely missing data, no fake 0)."""
    if x is None:
        return "—"
    if isinstance(x, float) and x != x:  # NaN
        return "—"
    return f"{x}{suffix}"


def cell_flag(cell: Dict[str, Any]) -> str:
    """Render audit-finding flag(s) inline next to a cell's row label."""
    audit = cell.get("audit", [])
    if not audit:
        return ""
    counts = {"error": 0, "warn": 0, "info": 0}
    for f in audit:
        counts[f.get("severity", "warn")] = counts.get(f.get("severity","warn"), 0) + 1
    out = []
    for sev, n in counts.items():
        if n:
            sym = {"error": "✗", "warn": "⚠", "info": "ⓘ"}[sev]
            out.append(f'<span class="flag flag-{sev}" title="see Data Integrity Audit section">{sym} {n}</span>')
    return "".join(out)


def html_main_latency_table(cells: List[Dict]) -> str:
    rows = []
    for c in cells:
        if c["is_isolated"] or not c["stats"] or not c["is_complete"]:
            continue
        s = c["stats"]
        tag = MODE_TAG.get(c["mode"], "")
        rows.append((c["mode"], c["bucket"], c["name"], s, tag))
    rows.sort(key=lambda x: (MODE_ORDER.get(x[0], 99), BUCKET_ORDER.get(x[1], 99)))
    if not rows:
        return ""
    cell_by_mb = {(c["mode"], c["bucket"]): c for c in cells if not c["is_isolated"]}
    any_streaming = any(s.get("is_streaming") for _, _, _, s, _ in rows)

    hdr = ('<table><thead><tr><th>Cell</th><th>n</th>'
           '<th>e2e min</th><th>e2e median</th><th>e2e max</th><th>e2e mean</th>'
           '<th>frontend<br>overhead min</th><th>frontend<br>overhead median</th>'
           '<th title="frontend_overhead_ms = e2e_ms − model_backend_http_ms (derived from two real measurements)">ⓘ</th>'
           '<th>backend min</th><th>backend median</th><th>backend max</th>'
           '<th>backend<br>ms/tok<br>est (median)</th>')
    if any_streaming:
        hdr += ('<th title="Real TTFT: t_first_SSE_chunk − t_request_start (streaming only)">TTFT<br>median</th>'
                '<th title="generation_ms: t_final_chunk − t_first_chunk (streaming only)">gen_ms<br>median</th>'
                '<th title="stream_inter_chunk_ms = generation_ms / (n_chunks−1). Mean time between SSE chunks. NOT per-token unless chunk==token.">inter-chunk<br>ms median<br><span style=\'font-size:9px;color:#d29922\'>≠ TPOT</span></th>'
                '<th title="True TPOT = generation_ms / (usage.completion_tokens − 1). Only set when vLLM sends usage in stream.">TPOT<br>median<br><span style=\'font-size:9px;color:#3fb950\'>(true)</span></th>')
    hdr += '<th>EOS%</th><th>route</th><th>cache hit%</th></tr></thead><tbody>'
    out = [hdr]

    for mode, bucket, name, s, tag in rows:
        route = list(s.get("routes", {}).keys())[0] if s.get("routes") else "—"
        c_obj = cell_by_mb.get((mode, bucket))
        flag = cell_flag(c_obj) if c_obj else ""
        row_html = (f'<tr><td><span class="tag {tag}">{mode}</span>{bucket}{flag}</td>'
                    f'<td>{s["n"]}</td>'
                    f'<td>{s["e2e_min"]}</td><td>{s["e2e_p50"]}</td><td>{s["e2e_max"]}</td>'
                    f'<td>{s["e2e_mean"]}</td>'
                    f'<td>{s["fe_min"]}</td><td>{s["fe_p50"]}</td><td></td>'
                    f'<td>{s["mb_min"]}</td><td>{s["mb_p50"]}</td><td>{s["mb_max"]}</td>'
                    f'<td>{fmt_v(s.get("tpot_p50"))}</td>')
        if any_streaming:
            if s.get("is_streaming"):
                row_html += (f'<td>{fmt_v(s.get("real_ttft_p50"))}</td>'
                             f'<td>{fmt_v(s.get("gen_ms_p50"))}</td>'
                             f'<td>{fmt_v(s.get("ichunk_p50"))}</td>'
                             f'<td>{fmt_v(s.get("real_tpot_p50"))}</td>')
            else:
                row_html += '<td>—</td><td>—</td><td>—</td><td>—</td>'
        row_html += (f'<td>{s["eos_pct"]}%</td><td><code>{route}</code></td>'
                     f'<td>{s["cache_hit_pct"]}%</td></tr>')
        out.append(row_html)
    out.append("</tbody></table>")
    if any_streaming:
        out.append('<p class="muted" style="font-size:11px;margin:4px 0">'
                   '<strong>TTFT</strong>: t_first_SSE_chunk − t_request_start (real, client-side). '
                   '<strong>gen_ms</strong>: t_final_chunk − t_first_chunk (total decode window). '
                   '<strong>inter-chunk</strong>: mean time between SSE chunks — NOT per-token unless vLLM sends one token per chunk. '
                   '<strong>TPOT (true)</strong>: gen_ms / (usage.completion_tokens−1) — empty when vLLM did not send usage in stream.</p>')
    return "\n".join(out)


def html_request_payload_table(cells: List[Dict]) -> str:
    """Show measured input/output token counts and RAG retrieval quality per cell."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    out = ['<table><thead><tr><th>Cell</th>'
           '<th>query<br>tokens</th><th>orig prompt<br>tokens</th>'
           '<th>aug prompt<br>tokens</th><th>RAG inflation<br>(aug/orig)</th>'
           '<th>n_output<br>tokens (mean)</th>'
           '<th>RAG chunks<br>(mean)</th><th>RAG top-score<br>(mean)</th>'
           '<th>cache write<br>ms (mean)</th></tr></thead><tbody>']
    for c in rows:
        s = c["stats"]
        tag = MODE_TAG.get(c["mode"], "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{fmt_v(s.get("query_words_mean"))}</td>'
                   f'<td>{fmt_v(s.get("orig_tokens_mean"))}</td>'
                   f'<td>{fmt_v(s.get("aug_tokens_mean"))}</td>'
                   f'<td>{fmt_v(s.get("prompt_inflation"), "×")}</td>'
                   f'<td>{fmt_v(s.get("n_out_mean"))}</td>'
                   f'<td>{fmt_v(s.get("rag_chunks_mean"))}</td>'
                   f'<td>{fmt_v(s.get("rag_score_mean"))}</td>'
                   f'<td>{fmt_v(s.get("cache_write_mean"))}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_tail_latency_table(cells: List[Dict]) -> str:
    """Per-component tail latency (= max for n<20). From CSV directly."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    any_large = any(c["stats"].get("n", 0) >= 20 for c in rows)
    stat_label = "p95" if any_large else "max"
    out = [f'<table><thead><tr><th>Cell</th>'
           f'<th>rag_embed<br>{stat_label}</th><th>rag_milvus<br>{stat_label}</th><th>rag_seaweed<br>{stat_label}</th>'
           f'<th>cache_embed<br>{stat_label}</th><th>cache_milvus<br>{stat_label}</th><th>cache_mongo<br>{stat_label}</th>'
           '</tr></thead><tbody>']
    for c in rows:
        s = c["stats"]
        tag = MODE_TAG.get(c["mode"], "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{fmt_v(s.get("rag_embed_p95"))}</td>'
                   f'<td>{fmt_v(s.get("rag_milvus_p95"))}</td>'
                   f'<td>{fmt_v(s.get("rag_seaweed_p95"))}</td>'
                   f'<td>{fmt_v(s.get("cache_embed_p95"))}</td>'
                   f'<td>{fmt_v(s.get("cache_milvus_p95"))}</td>'
                   f'<td>{fmt_v(s.get("cache_mongo_p95"))}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_pipeline_breakdown_table(cells: List[Dict]) -> str:
    rows = []
    for c in cells:
        if c["is_isolated"] or not c["stats"] or not c["is_complete"]:
            continue
        rows.append(c)
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    out = ['<table><thead><tr><th>Cell</th>'
           '<th>rag_embed</th><th>rag_milvus</th><th>rag_seaweed</th>'
           '<th>rag_retrieve<br>(combined)</th><th>rag_format</th>'
           '<th>cache_embed</th><th>cache_milvus</th><th>cache_mongo</th>'
           '<th>cache_write</th><th>sc_lookup</th><th>backend</th></tr></thead><tbody>']
    for c in rows:
        s = c["stats"]
        tag = MODE_TAG.get(c["mode"], "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{s["rag_embed_mean"]:.1f}</td><td>{s["rag_milvus_mean"]:.1f}</td>'
                   f'<td>{s["rag_seaweed_mean"]:.1f}</td>'
                   f'<td>{fmt_v(s.get("rag_retrieve_mean"))}</td>'
                   f'<td>{fmt_v(s.get("rag_format_mean"))}</td>'
                   f'<td>{s["cache_embed_mean"]:.1f}</td><td>{s["cache_milvus_mean"]:.1f}</td>'
                   f'<td>{s["cache_mongo_mean"]:.1f}</td>'
                   f'<td>{fmt_v(s.get("cache_write_mean"))}</td>'
                   f'<td>{s["sc_lookup_mean"]:.1f}</td><td>{s["mb_mean"]:.0f}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_cross_pod_table(cell: Dict[str, Any]) -> str:
    """Cross-pod comparison table for a single cell.

    Hardware pods (c7i bare-metal): IPC, ILP, MLP, AMAT, cache MPKI, DRAM.
    Software-only pods (vLLM on Nitro/GPU node): task-clock, ctx-switches,
    cpu-migrations, page-faults — hardware PMU not exposed by hypervisor.
    llmd_gateway (istio/envoy) runs on c7i so it gets full hardware counters.
    """
    per_pod = cell.get("per_pod_hw", {})
    if not per_pod:
        return ""

    hw_pods = [(k, per_pod[k]) for k in ALL_POD_KEYS
               if k in per_pod and k not in _SW_ONLY_PODS and per_pod[k].get("ipc")]
    sw_pods = [(k, per_pod[k]) for k in ALL_POD_KEYS
               if k in per_pod and k in _SW_ONLY_PODS and per_pod[k].get("task_clock_sec") is not None]

    if not hw_pods and not sw_pods:
        return ""

    out = ['<h4 style="color:var(--text-dim);font-size:13px;margin:12px 0 6px">Per-pod hardware counters (same query traffic)</h4>']

    if hw_pods:
        out.append('<p class="muted" style="font-size:11px;margin:0 0 6px">'
                   'DRAM GB/s is a node-wide IMC measurement — identical across pods by design. '
                   '<b>†</b> MPKI is cross-pass (instructions from pass1, misses from pass2).</p>')
        out.append('<table><thead><tr><th>Pod</th>'
                   '<th>IPC</th><th>ILP</th><th>MLP</th>'
                   '<th>L1 MPKI†</th><th>LLC MPKI†</th>'
                   '<th>mem_stall%</th><th>eff LLC lat (cy)</th>'
                   '<th>ctx-sw</th><th>cpu-mig</th>'
                   '<th>DRAM avg GB/s</th>'
                   '</tr></thead><tbody>')
        for pod_key, h in hw_pods:
            out.append(f'<tr><td><code>{pod_key}</code></td>'
                       f'<td>{fmt_v(h.get("ipc"))}</td>'
                       f'<td>{fmt_v(h.get("ilp"))}</td>'
                       f'<td>{fmt_v(h.get("mlp"))}</td>'
                       f'<td>{fmt_v(h.get("l1_mpki"))}</td>'
                       f'<td>{fmt_v(h.get("llc_mpki"))}</td>'
                       f'<td>{fmt_v(h.get("mem_stall_pct"), "%")}</td>'
                       f'<td>{fmt_v(h.get("eff_llc_lat_cy"))}</td>'
                       f'<td>{fmt_int(h.get("ctx_switches"))}</td>'
                       f'<td>{fmt_int(h.get("cpu_migrations"))}</td>'
                       f'<td>{fmt_v(h.get("dram_avg_total_gbs"))}</td></tr>')
        out.append('</tbody></table>')

    if sw_pods:
        out.append('<h4 style="color:var(--text-dim);font-size:13px;margin:16px 0 4px">'
                   'vLLM — software counters only (Nitro hypervisor, no hardware PMU)</h4>')
        out.append('<p class="muted" style="font-size:11px;margin:0 0 6px">'
                   'The GPU node runs on AWS Nitro which does not expose the hardware PMU to guests. '
                   'Only OS-level software events are available: CPU time consumed, scheduler preemptions, '
                   'migrations between cores, and page faults. These reflect the orchestration cost of '
                   'the vLLM Python process, not GPU compute.</p>')
        out.append('<table><thead><tr><th>Pod</th>'
                   '<th>task-clock (s)<br><span class="muted" style="font-size:10px">CPU time consumed</span></th>'
                   '<th>ctx-switches<br><span class="muted" style="font-size:10px">scheduler preemptions</span></th>'
                   '<th>cpu-migrations<br><span class="muted" style="font-size:10px">cross-core moves</span></th>'
                   '<th>page-faults<br><span class="muted" style="font-size:10px">minor+major</span></th>'
                   '</tr></thead><tbody>')
        for pod_key, h in sw_pods:
            out.append(f'<tr><td><code>{pod_key}</code></td>'
                       f'<td>{fmt_v(h.get("task_clock_sec"))}</td>'
                       f'<td>{fmt_int(h.get("ctx_switches"))}</td>'
                       f'<td>{fmt_int(h.get("cpu_migrations"))}</td>'
                       f'<td>{fmt_int(h.get("page_faults"))}</td></tr>')
        out.append('</tbody></table>')

    return "\n".join(out)


def html_tma_path_charts(per_pod_tma: Dict[str, Any]) -> str:
    """One stacked horizontal bar chart per path (rag/llm_direct/sc_a), pods on Y-axis."""
    tier_data = per_pod_tma.get("tok64", {})
    if not tier_data:
        return ""

    PATH_LABELS = {"rag": "RAG", "llm_direct": "LLM Direct", "sc_a": "Semantic Cache"}
    SEGMENTS = [
        ("retiring",              "Retiring",         "#2ea043"),
        ("fe_bound",              "Frontend Bound",   "#388bfd"),
        ("bad_spec",              "Bad Speculation",  "#e3b341"),
        ("toplev_memory_bound",   "BE / Memory",      "#f85149"),
        ("toplev_core_bound",     "BE / Core",        "#bc8cff"),
    ]
    # fallback: if toplev_memory_bound absent, use be_bound remainder
    out = ['<h2>TMA breakdown per path — averaged over query duration</h2>']
    out.append('<p class="muted">Top-level TMA slots split per pod. '
               'Retiring = useful work; Frontend Bound = instruction supply; '
               'Bad Speculation = branch/mispredict waste; BE/Memory = cache/DRAM stalls; '
               'BE/Core = execution-unit pressure. Values are % of pipeline slots.</p>')
    out.append('<div style="display:flex;flex-wrap:wrap;gap:24px;margin-bottom:32px">')

    chart_id = 0
    for path, path_label in PATH_LABELS.items():
        pods_data = tier_data.get(path, {})
        if not pods_data:
            continue
        pod_names = list(pods_data.keys())
        datasets = []
        for key, label, color in SEGMENTS:
            values = []
            for pod in pod_names:
                d = pods_data[pod]
                if key in d:
                    values.append(round(d[key], 1))
                elif key == "toplev_memory_bound":
                    # fallback: be_bound minus core if no memory split
                    be = d.get("be_bound", 0)
                    core = d.get("toplev_core_bound", 0)
                    values.append(round(max(be - core, 0), 1))
                elif key == "toplev_core_bound":
                    values.append(0.0)
                else:
                    values.append(0.0)
            datasets.append({"label": label, "data": values, "color": color})

        cid = f"tmaPath{chart_id}"
        chart_id += 1
        height = max(180, len(pod_names) * 38)
        out.append(f'<div style="flex:1;min-width:320px;max-width:560px">')
        out.append(f'<h4 style="margin:0 0 8px;font-size:13px">{path_label}</h4>')
        out.append(f'<canvas id="{cid}" height="{height}"></canvas>')
        out.append('</div>')

        ds_js = ",\n".join(
            f'{{label:{json.dumps(d["label"])},data:{json.dumps(d["data"])},'
            f'backgroundColor:{json.dumps(d["color"]+"cc")},borderColor:{json.dumps(d["color"])},'
            f'borderWidth:1}}'
            for d in datasets
        )
        out.append(f'''<script>new Chart(document.getElementById({json.dumps(cid)}),{{
  type:'bar',
  data:{{labels:{json.dumps(pod_names)},datasets:[{ds_js}]}},
  options:{{indexAxis:'y',responsive:true,plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:11}}}}}},
    tooltip:{{callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.x.toFixed(1)+'%'}}}}}},
    scales:{{x:{{stacked:true,max:100,title:{{display:true,text:'% pipeline slots'}}}},
             y:{{stacked:true}}}}}}}});</script>''')

    out.append('</div>')
    return "\n".join(out)


def html_tma_narrative_section(cell: Dict[str, Any], per_pod_tma_mode: Dict[str, Any]) -> str:
    """Render TMA narrative per pod for a single cell × mode."""
    if not per_pod_tma_mode:
        return ""
    per_pod_hw = cell.get("per_pod_hw", {})
    out = ['<h4 style="color:var(--text-dim);font-size:13px;margin:12px 0 6px">TMA interpretation per pod</h4>']
    out.append('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px;margin:8px 0">')
    for pod_key, tma_fields in per_pod_tma_mode.items():
        if not tma_fields:
            continue
        hw = per_pod_hw.get(pod_key, {})
        narrative = generate_tma_narrative(tma_fields, pod_key, cell.get("mode", ""), hw)
        # TMA level-1 bar values
        be = tma_fields.get("be_bound", tma_fields.get("toplev_be_bound"))
        fe = tma_fields.get("fe_bound", tma_fields.get("toplev_fe_bound"))
        re_ = tma_fields.get("retiring", tma_fields.get("toplev_retiring"))
        bs = tma_fields.get("bad_spec", tma_fields.get("toplev_bad_spec"))
        bar_html = ""
        if all(v is not None for v in (be, fe, re_, bs)):
            bar_html = (f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0">'
                        f'<div style="width:{re_}%;background:#3fb950" title="Retiring {re_}%"></div>'
                        f'<div style="width:{fe}%;background:#58a6ff" title="FE {fe}%"></div>'
                        f'<div style="width:{bs}%;background:#d29922" title="Bad-Spec {bs}%"></div>'
                        f'<div style="width:{be}%;background:#f85149" title="BE {be}%"></div>'
                        f'</div>'
                        f'<div style="font-size:10px;color:var(--text-dim)">Retiring {fmt_v(re_)}% · FE {fmt_v(fe)}% · Bad-Spec {fmt_v(bs)}% · BE {fmt_v(be)}%</div>')
        out.append(f'<div style="background:var(--panel-2);border:1px solid var(--border);border-radius:6px;padding:12px">'
                   f'<div style="font-size:12px;font-weight:600;color:var(--text)">{pod_key}</div>'
                   f'{bar_html}'
                   f'<div style="font-size:12px;color:var(--text-dim);margin-top:6px;line-height:1.5">{narrative}</div>'
                   f'</div>')
    out.append('</div>')
    return "\n".join(out)


def html_audit_section(run: Dict[str, Any]) -> str:
    """Render the data-integrity audit table covering every cell across all tiers."""
    rows: List[Tuple[str, str, Dict]] = []
    counts = {"error": 0, "warn": 0, "info": 0, "clean": 0}
    cross = cross_cell_audit(run["cells"], run.get("stream", {}), run.get("tma", {}))
    for f in cross:
        counts[f.get("severity","warn")] = counts.get(f.get("severity","warn"), 0) + 1
    for tier in ("tok64", "tok192", "tok320"):
        for c in run["cells"].get(tier, []):
            audit = c.get("audit", [])
            if not audit:
                counts["clean"] += 1
            else:
                for f in audit:
                    counts[f.get("severity","warn")] = counts.get(f.get("severity","warn"), 0) + 1
            rows.append((tier, c["name"], c))
    out = ['<h2>Data integrity audit</h2>']
    summary_bits = []
    if counts["clean"]:
        summary_bits.append(f'<strong style="color:var(--good)">{counts["clean"]} cells clean</strong>')
    if counts["error"]:
        summary_bits.append(f'<strong class="audit-error">{counts["error"]} ERRORs</strong>')
    if counts["warn"]:
        summary_bits.append(f'<strong class="audit-warn">{counts["warn"]} warnings</strong>')
    if counts["info"]:
        summary_bits.append(f'<strong class="audit-info">{counts["info"]} info</strong>')
    out.append(f'<p>{" · ".join(summary_bits)}</p>')
    out.append('<p class="muted">Each cell is checked for: file completeness (perf passes, CSV row count), '
               'mode-semantic consistency (route ↔ mode, bypass flags ↔ mode, telemetry zero/non-zero ↔ mode), '
               'mathematical sanity (latency monotonicity, e2e ≥ sub-phases, DRAM peak ≥ avg, L1 ≥ L2 ≥ L3 pyramid), '
               'and counter-value plausibility (IPC, AVX%, stall ratios). '
               'Cells with findings are also flagged inline in the latency tables above with '
               '<span class="flag flag-error">✗</span>/<span class="flag flag-warn">⚠</span>/<span class="flag flag-info">ⓘ</span> badges.</p>')
    out.append('<table class="audit-table"><thead><tr><th>Cell</th><th>severity</th><th>finding &amp; remediation</th></tr></thead><tbody>')
    any_finding = False
    def _render_finding_cell(f: Dict[str, str]) -> str:
        body = f["msg"]
        if f.get("remediation"):
            body += f'<br><span class="muted"><strong>Remediate:</strong> {f["remediation"]}</span>'
        return body
    # Cross-cell findings first
    for f in cross:
        any_finding = True
        sev = f["severity"]
        sev_html = f'<span class="audit-{sev}">{sev.upper()}</span>'
        out.append(f'<tr><td><code>cross-cell</code></td>'
                   f'<td>{sev_html}</td>'
                   f'<td>{_render_finding_cell(f)}</td></tr>')
    for tier, name, c in rows:
        for f in c.get("audit", []):
            any_finding = True
            sev = f["severity"]
            sev_html = f'<span class="audit-{sev}">{sev.upper()}</span>'
            out.append(f'<tr><td><code>{tier}/{name}</code></td>'
                       f'<td>{sev_html}</td>'
                       f'<td>{_render_finding_cell(f)}</td></tr>')
    if not any_finding:
        out.append('<tr><td colspan="3" style="text-align:center;color:var(--good);padding:18px">All cells passed every integrity check.</td></tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


def html_isolated_table(cells: List[Dict]) -> str:
    rows = []
    for c in cells:
        if not c["is_isolated"] or not c["stats"]:
            continue
        rows.append(c)
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return "<p class='muted'>No isolated-cell data yet.</p>"
    any_large = any(not c["stats"].get("small_n", True) for c in rows)
    if any_large:
        hdr = ('<table><thead><tr><th>Cell</th><th>n</th>'
               '<th>min (ms)</th><th>median (ms)</th><th>p95 (ms)</th><th>p99 (ms)</th>'
               '<th>mean (ms)</th><th>max (ms)</th></tr></thead><tbody>')
    else:
        hdr = ('<table><thead><tr><th>Cell</th><th>n</th>'
               '<th>min (ms)</th><th>median (ms)</th><th>max (ms)</th>'
               '<th>mean (ms)</th></tr></thead><tbody>')
    out = [hdr]
    for c in rows:
        s = c["stats"]
        tag = MODE_TAG.get(c["mode"], "")
        if any_large:
            out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                       f'<td>{s["n"]}</td><td>{fmt_v(s.get("min"))}</td><td>{fmt_v(s.get("median"))}</td>'
                       f'<td>{fmt_v(s.get("p95"))}</td><td>{fmt_v(s.get("p99"))}</td>'
                       f'<td>{s["mean"]}</td><td>{s["max"]}</td></tr>')
        else:
            out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                       f'<td>{s["n"]}</td><td>{fmt_v(s.get("min"))}</td><td>{fmt_v(s.get("median"))}</td>'
                       f'<td>{s["max"]}</td><td>{s["mean"]}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_gpu_table(cells: List[Dict]) -> str:
    """GPU metrics table from nvidia-smi dmon + pmon collected during pass1 queries."""
    rows = [c for c in cells if not c["is_isolated"] and (c.get("gpu_dmon") or c.get("gpu_pmon"))]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return "<p class='muted'>No GPU monitoring data in this run (nvidia-smi not available or no GPU).</p>"

    def _v(val, suffix="%"):
        if val is None:
            return "<td>—</td>"
        return f"<td>{val:.1f}{suffix}</td>"

    out = ['<table><thead><tr>'
           '<th>Cell</th>'
           '<th title="nvidia-smi dmon: basic GPU utilisation %">GPU util%</th>'
           '<th title="GPM SM Activity: fraction of SMs with active warps — drops to 0 between requests">SM activity%</th>'
           '<th title="GPM HMMA Tensor: BF16/FP16 tensor core utilisation — should be high during attention+FFN">HMMA tensor%</th>'
           '<th title="GPM DRAM Activity: HBM access fraction — high = memory-bandwidth bound">DRAM activity%</th>'
           '<th title="Per-process SM% (vLLM only, from pmon)">vLLM SM%</th>'
           '<th title="vLLM VRAM usage in MB (from pmon)">vLLM VRAM (MB)</th>'
           '</tr></thead><tbody>']

    for c in rows:
        tag = MODE_TAG.get(c["mode"], "")
        label = f'<span class="tag {tag}">{c["mode"]}</span> {c["bucket"]}'
        d = c.get("gpu_dmon", {})
        p = c.get("gpu_pmon", {})

        # Find compute process from pmon (type=C, largest FB = vLLM)
        vllm_sm = vllm_fb = None
        if p:
            compute = {pid: v for pid, v in p.items() if v.get("fb_mb")}
            if compute:
                vllm_pid = max(compute, key=lambda pid: compute[pid].get("fb_mb") or 0)
                vllm_sm = compute[vllm_pid].get("sm_pct")
                vllm_fb = compute[vllm_pid].get("fb_mb")

        out.append(f'<tr><td>{label}</td>'
                   + _v(d.get("sm_pct"))
                   + _v(d.get("sm_activity_pct"))
                   + _v(d.get("hmma_tensor_pct"))
                   + _v(d.get("dram_activity_pct"))
                   + _v(vllm_sm)
                   + (f'<td>{int(vllm_fb):,}</td>' if vllm_fb else '<td>—</td>')
                   + '</tr>')

    out.append("</tbody></table>")
    return "\n".join(out)


def html_vllm_metrics_table(cells: List[Dict]) -> str:
    """vLLM engine internals scraped from /metrics during Pass 1.

    Gauges as min/median/max (low-n convention, matching latency tables);
    counters as the delta over the window (+ per-second rate).
    """
    rows = [c for c in cells if not c["is_isolated"] and c.get("vllm")
            and (c["vllm"].get("gauges") or c["vllm"].get("counters"))]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""

    def _g(c, name):
        return c["vllm"].get("gauges", {}).get(name)

    def _kv(c):  # KV-cache gauge under either modern/older name
        g = c["vllm"].get("gauges", {})
        return g.get("vllm:kv_cache_usage_perc") or g.get("vllm:gpu_cache_usage_perc")

    def _ctr(c, name):
        return c["vllm"].get("counters", {}).get(name)

    def _mmm(d, scale=1.0, suffix=""):
        if not d:
            return "—"
        return (f'{d["min"]*scale:.1f} / {d["median"]*scale:.1f} / '
                f'{d["max"]*scale:.1f}{suffix}')

    out = ['<p class="muted" style="font-size:11px;margin:0 0 6px">'
           'Scraped from vLLM <code>/metrics</code> during the Pass-1 window. '
           'Gauges shown as <strong>min / median / max</strong> over the poll samples '
           '(low-n convention); counters as the delta over the window. '
           'These are engine internals that perf / nsys / ncu cannot see.</p>',
           '<table><thead><tr><th>Cell</th>'
           '<th>requests running<br><span class="muted" style="font-size:10px">min/med/max</span></th>'
           '<th>requests waiting<br><span class="muted" style="font-size:10px">min/med/max</span></th>'
           '<th>KV-cache %<br><span class="muted" style="font-size:10px">min/med/max</span></th>'
           '<th>preemptions<br><span class="muted" style="font-size:10px">Δ window</span></th>'
           '<th>gen tok/s</th><th>prompt tok/s</th>'
           '<th>polls</th></tr></thead><tbody>']
    for c in rows:
        tag = MODE_TAG.get(c["mode"], "")
        gen = _ctr(c, "vllm:generation_tokens_total")
        prm = _ctr(c, "vllm:prompt_tokens_total")
        pre = _ctr(c, "vllm:num_preemptions_total")
        n_polls = c["vllm"].get("n_polls", "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{_mmm(_g(c, "vllm:num_requests_running"))}</td>'
                   f'<td>{_mmm(_g(c, "vllm:num_requests_waiting"))}</td>'
                   f'<td>{_mmm(_kv(c), scale=100, suffix="%")}</td>'
                   f'<td>{fmt_int(pre["delta"]) if pre else "—"}</td>'
                   f'<td>{fmt_v(gen["rate_per_s"]) if gen else "—"}</td>'
                   f'<td>{fmt_v(prm["rate_per_s"]) if prm else "—"}</td>'
                   f'<td>{n_polls}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_hw_table(cells: List[Dict]) -> str:
    rows = []
    for c in cells:
        if c["is_isolated"]:
            continue
        if not c["hw"].get("ipc"):
            continue
        if not c["is_complete"]:
            continue
        rows.append(c)
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    out = ['<table><thead><tr><th>Cell</th>'
           '<th>IPC</th><th>ILP</th><th>MLP</th>'
           '<th>avg freq<br>GHz</th>'
           '<th>L1 MPKI</th><th>L2 MPKI</th><th>LLC MPKI</th>'
           '<th>LLC miss<br>rate %</th>'
           '<th>AVX-512<br>FP32%</th>'
           '<th>real DRAM avg<br>(GB/s)</th><th>peak read<br>(GB/s)</th>'
           '<th>mem-stall%</th>'
           '<th>arith intensity<br>(FLOPs/byte)</th>'
           '</tr></thead><tbody>']
    for c in rows:
        h = c["hw"]
        tag = MODE_TAG.get(c["mode"], "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{fmt_v(h.get("ipc"))}</td>'
                   f'<td>{fmt_v(h.get("ilp"))}</td>'
                   f'<td>{fmt_v(h.get("mlp"))}</td>'
                   f'<td>{fmt_v(h.get("avg_freq_ghz"))}</td>'
                   f'<td>{fmt_v(h.get("l1_mpki"))}</td><td>{fmt_v(h.get("l2_mpki"))}</td><td>{fmt_v(h.get("llc_mpki"))}</td>'
                   f'<td>{fmt_v(h.get("llc_miss_rate_pct"), "%")}</td>'
                   f'<td>{fmt_v(h.get("avx512_fp32_pct"))}</td>'
                   f'<td>{fmt_v(h.get("dram_avg_total_gbs"))}</td>'
                   f'<td>{fmt_v(h.get("dram_peak_read_gbs"))}</td>'
                   f'<td>{fmt_v(h.get("mem_stall_pct"))}</td>'
                   f'<td>{fmt_v(h.get("arith_intensity_fp32"))}</td>'
                   f'</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_bandwidth_table(cells: List[Dict], stream: Dict[str, float]) -> str:
    """Detailed DRAM bandwidth: peak/avg read+write, totals, wall, %ceiling."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and (c["hw"].get("dram_peak_read_gbs") or 0) > 0]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    triad = stream.get("triad_gbs") if stream else None
    pct_header = '<th>% of STREAM<br>Triad</th>' if triad else '<th>% of STREAM<br>Triad</th>'
    out = ['<p class="muted" style="font-size:11px;margin:0 0 6px">'
           'Peak read/write are only meaningful when pass3 runs with <code>--interval-print</code> '
           '(EKS bare-metal). On local runs a single total is recorded — peak = avg = total/wall_sec. '
           'Burst factor = peak/avg (1.0 on local single-measurement runs).</p>',
           '<table><thead><tr><th>Cell</th>'
           '<th>peak read<br>GB/s</th><th>peak write<br>GB/s</th>'
           '<th>avg read<br>GB/s</th><th>avg write<br>GB/s</th>'
           '<th>avg total<br>GB/s</th>'
           f'{pct_header}'
           '<th>read GiB<br>total</th><th>write GiB<br>total</th>'
           '<th>R:W ratio</th><th>burst<br>(peak/avg)</th><th>wall sec</th></tr></thead><tbody>']
    for c in rows:
        h = c["hw"]
        tag = MODE_TAG.get(c["mode"], "")
        avg_total = h.get("dram_avg_total_gbs")
        pct_ceiling = round(avg_total / triad * 100, 2) if (triad and avg_total is not None) else None
        intervals = h.get("dram_intervals", 0) or 0
        # Peak == avg when no interval-print data (single measurement) — show — to avoid false precision
        peak_r = fmt_v(h.get("dram_peak_read_gbs"))  if intervals > 0 else "≈avg"
        peak_w = fmt_v(h.get("dram_peak_write_gbs")) if intervals > 0 else "≈avg"
        burst   = fmt_v(h.get("dram_burst_factor"), "×") if intervals > 0 else "—"
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{peak_r}</td><td>{peak_w}</td>'
                   f'<td>{fmt_v(h.get("dram_avg_read_gbs"))}</td><td>{fmt_v(h.get("dram_avg_write_gbs"))}</td>'
                   f'<td>{fmt_v(avg_total)}</td>'
                   f'<td>{fmt_v(pct_ceiling, "%")}</td>'
                   f'<td>{fmt_v(h.get("dram_total_read_gib"))}</td><td>{fmt_v(h.get("dram_total_write_gib"))}</td>'
                   f'<td>{fmt_v(h.get("dram_rw_ratio"))}</td><td>{burst}</td>'
                   f'<td>{fmt_v(h.get("dram_wall_sec"))}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_stalls_table(cells: List[Dict]) -> str:
    """Stall breakdown: how much of CPU time is wasted, and on what."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["hw"].get("ipc")]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    out = ['<table><thead><tr><th>Cell</th>'
           '<th>stalls (total)<br>% of cycles</th>'
           '<th>stalls (LLC-miss)<br>% of cycles</th>'
           '<th>LLC-miss share<br>of stalls</th>'
           '<th>load-bound<br>% of cycles</th>'
           '<th>spec waste<br>% uops</th>'
           '<th>branch-miss<br>per kins</th>'
           '<th>ctx-sw</th><th>cpu-mig</th></tr></thead><tbody>']
    for c in rows:
        h = c["hw"]
        tag = MODE_TAG.get(c["mode"], "")
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{fmt_v(h.get("stalls_tot_pct"))}</td>'
                   f'<td>{fmt_v(h.get("stalls_l3_pct"))}</td>'
                   f'<td>{fmt_v(h.get("stalls_l3_share"), "%")}</td>'
                   f'<td>{fmt_v(h.get("mem_stall_pct"))}</td>'
                   f'<td>{fmt_v(h.get("spec_waste_pct"))}</td>'
                   f'<td>{fmt_v(h.get("brmis_ki"))}</td>'
                   f'<td>{fmt_int(h.get("ctx_switches"))}</td>'
                   f'<td>{fmt_int(h.get("cpu_migrations"))}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def html_prefetch_table(cells: List[Dict]) -> str:
    """HW prefetcher effectiveness."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and (c["hw"].get("hwpf_total") or 0) > 0]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    out = ['<table><thead><tr><th>Cell</th>'
           '<th>hwpf total</th><th>hwpf miss</th>'
           '<th>hwpf miss %</th></tr></thead><tbody>']
    for c in rows:
        h = c["hw"]
        tag = MODE_TAG.get(c["mode"], "")
        total = h.get("hwpf_total")
        miss_pct = h.get("hwpf_miss_pct")
        miss_count = int(total * miss_pct / 100) if (total is not None and miss_pct is not None) else None
        out.append(f'<tr><td><span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}{cell_flag(c)}</td>'
                   f'<td>{fmt_int(total)}</td>'
                   f'<td>{fmt_int(miss_count)}</td>'
                   f'<td>{fmt_v(miss_pct, "%")}</td></tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def chart_data_latency(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell e2e latency bar chart. Uses median/max/mean — p95/p99 are meaningless for n<20."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "median": [c["stats"]["e2e_p50"]  for c in rows],
        "max":    [c["stats"]["e2e_max"]  for c in rows],
        "mean":   [c["stats"]["e2e_mean"] for c in rows],
    }


def chart_data_pipeline(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell stacked bar of pipeline component means."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "rag_embed":   [c["stats"]["rag_embed_mean"] for c in rows],
        "rag_milvus":  [c["stats"]["rag_milvus_mean"] for c in rows],
        "rag_seaweed": [c["stats"]["rag_seaweed_mean"] for c in rows],
        "cache_lookup":[c["stats"]["sc_lookup_mean"] for c in rows],
        "backend":     [c["stats"]["mb_mean"] for c in rows],
    }


def chart_data_pod_breakdown(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell stacked bar decomposing e2e mean into pod/service contributions."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    labels, vllm, embed, milvus, seaweed, cache, other = [], [], [], [], [], [], []
    for c in rows:
        s = c["stats"]
        e2e     = s["e2e_mean"]
        _vllm   = s["mb_mean"]
        _embed  = s["rag_embed_mean"]
        _milvus = s["rag_milvus_mean"]
        _sweed  = s["rag_seaweed_mean"]
        _cache  = s["sc_lookup_mean"]
        _other  = max(0.0, e2e - _vllm - _embed - _milvus - _sweed - _cache)
        labels.append(f'{c["mode"]} {c["bucket"]}')
        vllm.append(round(_vllm, 1))
        embed.append(round(_embed, 1))
        milvus.append(round(_milvus, 1))
        seaweed.append(round(_sweed, 1))
        cache.append(round(_cache, 1))
        other.append(round(_other, 1))
    return {"labels": labels, "vllm": vllm, "embed": embed,
            "milvus": milvus, "seaweed": seaweed, "cache": cache, "other": other}


def chart_data_payload(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell input/output token sizes — stacked bar."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]
            and c["stats"].get("orig_tokens_mean") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    out = {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "orig": [c["stats"]["orig_tokens_mean"] for c in rows],
        "rag_inject": [(c["stats"].get("aug_tokens_mean") or c["stats"]["orig_tokens_mean"])
                       - c["stats"]["orig_tokens_mean"] for c in rows],
        "n_out": [c["stats"]["n_out_mean"] for c in rows],
    }
    return out


def chart_data_hw_counters(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell IPC + average frequency."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("ipc") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "ipc": [c["hw"]["ipc"] for c in rows],
        "freq": [c["hw"]["avg_freq_ghz"] for c in rows],
        "l1_mpki": [c["hw"]["l1_mpki"] for c in rows],
        "llc_miss": [c["hw"].get("llc_miss_rate_pct") for c in rows],
    }


def chart_data_stalls(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell stall fractions — bar chart (does not stack since values can exceed 100% via cross-pass)."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("stalls_tot_pct") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "stalls_tot": [c["hw"].get("stalls_tot_pct") for c in rows],
        "mem_stall":  [c["hw"].get("mem_stall_pct") for c in rows],
        "spec_waste": [c["hw"].get("spec_waste_pct") for c in rows],
    }


def chart_data_prefetch(cells: List[Dict]) -> Dict[str, Any]:
    """HW prefetcher miss% bar."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("hwpf_miss_pct") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "miss_pct": [c["hw"]["hwpf_miss_pct"] for c in rows],
    }


def chart_data_isolated(cells: List[Dict]) -> Dict[str, Any]:
    """bge_isolated and hnsw_isolated: mean/median/max (p95 only if n≥20)."""
    rows = [c for c in cells if c["is_isolated"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "mean":   [c["stats"]["mean"]             for c in rows],
        "median": [c["stats"]["median"]            for c in rows],
        "max":    [c["stats"]["max"]               for c in rows],
    }


def chart_data_tail_latency(cells: List[Dict]) -> Dict[str, Any]:
    """Pipeline-component p95 stacked bar."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "rag_embed_p95":   [c["stats"].get("rag_embed_p95") or 0 for c in rows],
        "rag_milvus_p95":  [c["stats"].get("rag_milvus_p95") or 0 for c in rows],
        "rag_seaweed_p95": [c["stats"].get("rag_seaweed_p95") or 0 for c in rows],
        "cache_embed_p95":  [c["stats"].get("cache_embed_p95") or 0 for c in rows],
        "cache_milvus_p95": [c["stats"].get("cache_milvus_p95") or 0 for c in rows],
        "cache_mongo_p95":  [c["stats"].get("cache_mongo_p95") or 0 for c in rows],
    }


def chart_data_dram(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell real DRAM BW (peak read + write). Excludes cells without pass3 data."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("dram_peak_read_gbs") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "peak_read":  [c["hw"]["dram_peak_read_gbs"] for c in rows],
        "peak_write": [c["hw"]["dram_peak_write_gbs"] for c in rows],
        "avg_read":   [c["hw"]["dram_avg_read_gbs"] for c in rows],
        "avg_write":  [c["hw"]["dram_avg_write_gbs"] for c in rows],
    }


def chart_data_pyramid(cells: List[Dict]) -> Dict[str, Any]:
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("l1_hits") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "l1": [c["hw"]["l1_hits"] for c in rows],
        "l2": [c["hw"]["l2_hits"] for c in rows],
        "l3": [c["hw"]["l3_hits"] for c in rows],
    }


def chart_data_gpu(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell GPU utilisation: SM activity %, tensor-core (HMMA) %, HBM (DRAM) %."""
    rows = [c for c in cells if not c["is_isolated"] and c.get("gpu_dmon")]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    def _g(c, *keys):
        d = c.get("gpu_dmon", {})
        for k in keys:
            if d.get(k) is not None:
                return d[k]
        return None
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "sm":   [_g(c, "sm_activity_pct", "sm_pct") for c in rows],
        "hmma": [_g(c, "hmma_tensor_pct") for c in rows],
        "dram": [_g(c, "dram_activity_pct") for c in rows],
    }


def chart_data_decomposition(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell e2e time decomposition: stacked bar showing where time is spent.

    Segments (all in median ms):
      embed    — rag_embed_ms + cache_embed_ms
      milvus   — rag_milvus_ms + cache_milvus_ms
      seaweed  — rag_seaweed_ms
      mongo    — cache_mongo_ms
      vllm     — model_backend_http_ms
      remainder— frontend_overhead_ms - (embed+milvus+seaweed+mongo)
    """
    rows = [c for c in cells if not c["is_isolated"] and c.get("raw_rows")]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return {"labels": [], "embed": [], "milvus": [], "seaweed": [],
                "mongo": [], "vllm": [], "remainder": []}

    def _med(csv_rows, field):
        vals = [_f(r.get(field, 0)) for r in csv_rows if _f(r.get(field, 0)) > 0]
        return _pct(vals, 50) if vals else 0.0

    labels, embed, milvus, seaweed, mongo, vllm_seg, remainder = [], [], [], [], [], [], []
    for c in rows:
        ok_rows = [r for r in c["raw_rows"] if not r.get("error")]
        if not ok_rows:
            continue
        tag = f'{c["mode"]} {c["bucket"]}'
        e   = round(_med(ok_rows, "rag_embed_ms")   + _med(ok_rows, "cache_embed_ms"),  1)
        m   = round(_med(ok_rows, "rag_milvus_ms")  + _med(ok_rows, "cache_milvus_ms"), 1)
        sw  = round(_med(ok_rows, "rag_seaweed_ms"), 1)
        mg  = round(_med(ok_rows, "cache_mongo_ms"), 1)
        vl  = round(_med(ok_rows, "model_backend_http_ms"), 1)
        fe  = round(_med(ok_rows, "frontend_overhead_ms"), 1)
        rem = round(max(0.0, fe - e - m - sw - mg), 1)
        labels.append(tag)
        embed.append(e); milvus.append(m); seaweed.append(sw)
        mongo.append(mg); vllm_seg.append(vl); remainder.append(rem)

    return {"labels": labels, "embed": embed, "milvus": milvus,
            "seaweed": seaweed, "mongo": mongo, "vllm": vllm_seg, "remainder": remainder}


def chart_data_cpu_gpu_donut(cells: List[Dict]) -> Dict[str, Any]:
    """CPU vs GPU time split across all non-isolated cells with raw_rows.

    Returns {"cpu": ms, "gpu": ms} where:
      gpu = median model_backend_http_ms (time waiting for vLLM)
      cpu = median e2e_ms - gpu (all CPU-side work: RAG, embed, cache, FastAPI)
    Sums medians across all cells and returns the aggregate.
    Returns empty dict if all values are 0.
    """
    rows = [c for c in cells if not c["is_isolated"] and c.get("raw_rows")]
    if not rows:
        return {}

    def _med(csv_rows, field):
        vals = [_f(r.get(field, 0)) for r in csv_rows if _f(r.get(field, 0)) > 0]
        return _pct(vals, 50) if vals else 0.0

    total_cpu = total_gpu = 0.0
    for c in rows:
        ok_rows = [r for r in c["raw_rows"] if not r.get("error")]
        if not ok_rows:
            continue
        e2e = _med(ok_rows, "e2e_ms")
        gpu = _med(ok_rows, "model_backend_http_ms")
        cpu = max(0.0, e2e - gpu)
        total_gpu += round(gpu, 1)
        total_cpu += round(cpu, 1)

    result = {"cpu": round(total_cpu, 1), "gpu": round(total_gpu, 1)}
    if result["cpu"] == 0.0 and result["gpu"] == 0.0:
        return {}
    return result


def chart_data_bucket_components(cells: List[Dict]) -> Dict[str, Any]:
    """Per-mode, per-bucket component breakdown.

    Returns {mode: {bucket: {embed, milvus, seaweed, mongo, vllm, remainder}}}
    Only includes modes with at least one bucket of data.
    """
    rows = [c for c in cells if not c["is_isolated"] and c.get("raw_rows")]
    if not rows:
        return {}

    def _med(csv_rows, field):
        vals = [_f(r.get(field, 0)) for r in csv_rows if _f(r.get(field, 0)) > 0]
        return _pct(vals, 50) if vals else 0.0

    result: Dict[str, Any] = {}
    for c in rows:
        ok_rows = [r for r in c["raw_rows"] if not r.get("error")]
        if not ok_rows:
            continue
        mode   = c["mode"]
        bucket = c["bucket"]
        e   = round(_med(ok_rows, "rag_embed_ms")   + _med(ok_rows, "cache_embed_ms"),  1)
        m   = round(_med(ok_rows, "rag_milvus_ms")  + _med(ok_rows, "cache_milvus_ms"), 1)
        sw  = round(_med(ok_rows, "rag_seaweed_ms"), 1)
        mg  = round(_med(ok_rows, "cache_mongo_ms"), 1)
        vl  = round(_med(ok_rows, "model_backend_http_ms"), 1)
        fe  = round(_med(ok_rows, "frontend_overhead_ms"), 1)
        rem = round(max(0.0, fe - e - m - sw - mg), 1)
        result.setdefault(mode, {})[bucket] = {
            "embed": e, "milvus": m, "seaweed": sw,
            "mongo": mg, "vllm": vl, "remainder": rem,
        }
    return result


def chart_data_generation_by_bucket(cells: List[Dict]) -> Dict[str, Any]:
    """Decode latency + TTFT/frontend p50 per bucket for rag and llm_direct modes.

    Streaming runs: gen_ms (t_first→t_last chunk) + real_ttft (t_first chunk).
    Non-streaming runs: mb_p50 (model_backend_http_ms, full vLLM round-trip) +
                        fe_p50 (frontend_overhead_ms, FastAPI overhead proxy).
    """
    bucket_order = ['short', 'medium', 'long', 'very_long']
    result: Dict[str, Any] = {}
    for mode in ('rag', 'llm_direct'):
        mode_cells = [c for c in cells
                      if c['mode'] == mode and c['is_complete']
                      and c['stats'] and not c.get('is_isolated')]
        if len(mode_cells) < 2:
            continue
        by_bucket = {c['bucket']: c for c in mode_cells}
        buckets = [b for b in bucket_order if b in by_bucket]
        if len(buckets) < 2:
            continue
        is_streaming = any(by_bucket[b]['stats'].get('is_streaming') for b in buckets)
        if is_streaming:
            decode_vals = [round(by_bucket[b]['stats'].get('gen_ms_p50') or 0, 1) for b in buckets]
            ttft_vals   = [round(by_bucket[b]['stats'].get('real_ttft_p50') or 0, 1) for b in buckets]
            decode_label = 'gen_ms (decode window)'
            ttft_label   = 'TTFT (real)'
        else:
            decode_vals = [round(by_bucket[b]['stats'].get('mb_p50') or 0, 1) for b in buckets]
            ttft_vals   = [round(by_bucket[b]['stats'].get('fe_p50') or 0, 1) for b in buckets]
            decode_label = 'backend_ms (prefill+decode)'
            ttft_label   = 'frontend_overhead_ms'
        result[mode] = {
            'buckets':      buckets,
            'gen_ms':       decode_vals,
            'ttft':         ttft_vals,
            'is_streaming': is_streaming,
            'decode_label': decode_label,
            'ttft_label':   ttft_label,
        }
    return result


def chart_data_vllm(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell vLLM KV-cache usage (min/median/max %) scraped from /metrics."""
    rows = [c for c in cells if not c["is_isolated"] and c.get("vllm")
            and c["vllm"].get("gauges")]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    def _kv(c, stat):
        g = c["vllm"]["gauges"]
        d = g.get("vllm:kv_cache_usage_perc") or g.get("vllm:gpu_cache_usage_perc")
        return round(d[stat] * 100, 1) if d else None
    rows = [c for c in rows if _kv(c, "median") is not None]
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "kv_min": [_kv(c, "min") for c in rows],
        "kv_med": [_kv(c, "median") for c in rows],
        "kv_max": [_kv(c, "max") for c in rows],
    }


def chart_data_simd(cells: List[Dict]) -> Dict[str, Any]:
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["hw"].get("avx512_pct") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels": [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "avx512": [c["hw"]["avx512_pct"] for c in rows],
        "avx256": [c["hw"]["avx256_pct"] for c in rows],
    }


def chart_data_e2e_modes(cells: List[Dict]) -> Dict[str, Any]:
    """e2e p50 by mode × bucket for stacked bars (mode comparison)."""
    by_bucket: Dict[str, Dict[str, float]] = {}
    for c in cells:
        if c["is_isolated"] or not c["stats"]: continue
        by_bucket.setdefault(c["bucket"], {})[c["mode"]] = c["stats"]["e2e_p50"]
    bucket_order = sorted(by_bucket.keys(), key=lambda b: BUCKET_ORDER.get(b, 99))
    return {
        "labels": bucket_order,
        "rag":         [by_bucket[b].get("rag", 0) for b in bucket_order],
        "sc_a":        [by_bucket[b].get("sc_a", 0) for b in bucket_order],
        "sc_b":        [by_bucket[b].get("sc_b", 0) for b in bucket_order],
        "llm_direct":  [by_bucket[b].get("llm_direct", 0) for b in bucket_order],
    }


def chart_data_streaming_latency(cells: List[Dict]) -> Optional[Dict[str, Any]]:
    """TTFT, generation_ms, and real TPOT per cell — only for streaming runs."""
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"]
            and c["stats"] and c["stats"].get("is_streaming")]
    if not rows:
        return None
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    return {
        "labels":      [f'{c["mode"]} {c["bucket"]}' for c in rows],
        "ttft_p50":    [c["stats"].get("real_ttft_p50") or 0 for c in rows],
        "gen_ms_p50":  [c["stats"].get("gen_ms_p50") or 0 for c in rows],
        "tpot_p50":    [c["stats"].get("real_tpot_p50") or 0 for c in rows],
        "ichunk_p50":  [c["stats"].get("ichunk_p50") or 0 for c in rows],
    }


def chart_data_tma_tier(tma_tier: Dict[str, Any]) -> Dict[str, Any]:
    """Build TMA chart data from one tier's mode→{fields} dict.

    Only modes whose 4 categories were ALL successfully captured are plotted;
    modes with any missing slot are skipped (so the chart never fabricates 0%).
    """
    out = {"labels": [], "retiring": [], "fe_bound": [], "be_bound": [], "bad_spec": []}
    for mode in ("rag", "llm_direct", "sc_a"):
        t = tma_tier.get(mode, {})
        if not t or t.get("_failed"): continue
        # Try slots-derived first (perf stat topdown-*), fall back to toplev.py output
        re_v = t.get("retiring", t.get("toplev_retiring"))
        fe_v = t.get("fe_bound", t.get("toplev_fe_bound"))
        be_v = t.get("be_bound", t.get("toplev_be_bound"))
        bs_v = t.get("bad_spec", t.get("toplev_bad_spec"))
        if None in (re_v, fe_v, be_v, bs_v):
            continue
        out["labels"].append(mode)
        out["retiring"].append(re_v)
        out["fe_bound"].append(fe_v)
        out["be_bound"].append(be_v)
        out["bad_spec"].append(bs_v)
    return out


def chart_data_scaling(cells_by_tier: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """e2e p50 vs token tier (64/192/512) per mode (medium bucket)."""
    tiers = [t for t in ("tok64", "tok192", "tok320") if t in cells_by_tier]
    series: Dict[str, List[float]] = {m: [] for m in ("rag", "sc_a", "sc_b", "llm_direct")}
    for tier in tiers:
        bymode = {c["mode"]: c["stats"]["e2e_p50"]
                  for c in cells_by_tier[tier]
                  if not c["is_isolated"] and c["bucket"] == "medium" and c["stats"]}
        for m in series:
            series[m].append(bymode.get(m, 0))
    return {"labels": [t.replace("tok", "") for t in tiers], **series}


def chart_data_vllm_cpu(cells: List[Dict]) -> Dict[str, Any]:
    """Per-cell vLLM pod CPU utilization (software counters only — Nitro hypervisor).

    CPUs utilized = task_clock_sec / (n_requests * e2e_mean_ms / 1000).
    Also returns ctx_switches_per_req and page_faults_per_req.
    """
    rows = [c for c in cells if not c["is_isolated"] and c["is_complete"] and c["stats"]
            and c.get("per_pod_hw", {}).get("vllm", {}).get("task_clock_sec") is not None]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    labels, cpu_util, ctx_per_req, pf_per_req = [], [], [], []
    for c in rows:
        hw = c["per_pod_hw"]["vllm"]
        s = c["stats"]
        wall_sec = s["n"] * s["e2e_mean"] / 1000.0
        util = round(hw["task_clock_sec"] / wall_sec, 4) if wall_sec > 0 else 0
        ctx = round((hw.get("ctx_switches") or 0) / max(s["n"], 1), 1)
        pf  = round((hw.get("page_faults")  or 0) / max(s["n"], 1), 1)
        labels.append(f'{c["mode"]} {c["bucket"]}')
        cpu_util.append(util)
        ctx_per_req.append(ctx)
        pf_per_req.append(pf)
    return {"labels": labels, "cpu_util": cpu_util,
            "ctx_per_req": ctx_per_req, "pf_per_req": pf_per_req}


def html_per_request_timeline(cells: List[Dict]) -> str:
    rows = [c for c in cells if not c["is_isolated"] and c.get("raw_rows")]
    rows.sort(key=lambda c: (MODE_ORDER.get(c["mode"], 99), BUCKET_ORDER.get(c["bucket"], 99)))
    if not rows:
        return ""
    parts = ['<h3>Per-request timestamps &amp; nsys anchor</h3>',
             '<p style="font-size:12px;color:#555">Use the <strong>timestamp</strong> column to find '
             'the exact window in Jaeger or nsys-ui. Each row = one benchmark request.</p>']
    for c in rows:
        tag = MODE_TAG.get(c["mode"], "")
        label = f'<span class="tag {tag}">{c["mode"]}</span>{c["bucket"]}'
        rrows = c["raw_rows"]
        has_streaming = any(r.get("ttft_ms") for r in rrows)
        ttft_col_label = "ttft ms (real)" if has_streaming else "fe overhead ms"
        tbl = ['<table style="font-size:11px">',
               f'<thead><tr><th>#</th><th>timestamp (UTC)</th><th>request_id</th>'
               f'<th>e2e ms</th><th>backend ms</th><th>{ttft_col_label}</th>'
               f'<th>route</th><th>error</th></tr></thead><tbody>']
        for i, r in enumerate(rrows, 1):
            ts = str(r.get("timestamp") or "")
            rid = str(r.get("request_id") or "")[:8]
            e2e = _f(r.get("e2e_ms"))
            vllm = _f(r.get("model_backend_http_ms"))
            ttft_val = _f(r.get("ttft_ms")) if r.get("ttft_ms") else _f(r.get("frontend_overhead_ms"))
            route = str(r.get("route") or "")
            err = str(r.get("error") or "")
            err_style = ' style="color:red"' if err else ""
            tbl.append(f'<tr><td>{i}</td><td><code>{ts}</code></td><td><code>{rid}</code></td>'
                       f'<td>{e2e:.0f}</td><td>{vllm:.0f}</td><td>{ttft_val:.0f}</td>'
                       f'<td>{route}</td><td{err_style}>{err}</td></tr>')
        tbl.append('</tbody></table>')
        parts.append(f'<details><summary>{label} — {len(rrows)} requests</summary>')
        parts.append("\n".join(tbl))
        parts.append('</details>')
    return "\n".join(parts)


def render_html(run: Dict[str, Any]) -> str:
    cells_by_tier = run["cells"]
    all_cells_main = []
    for tier_cells in cells_by_tier.values():
        all_cells_main.extend(tier_cells)

    # Per-tier completion summary — use actual cell count, not a hardcoded expected value
    tier_status_html = []
    for tier in ("tok64", "tok192", "tok320"):
        tcells = cells_by_tier.get(tier, [])
        total = len(tcells)
        complete = sum(1 for c in tcells if c["perf_passes"] >= (1 if c["is_isolated"] else 4))
        tier_status_html.append(
            f'<div class="meta-item"><div class="k">{tier}</div>'
            f'<div class="v">{complete}/{total} cells complete</div></div>'
        )

    # Latest CSV mtime → freshness
    fresh = "—"
    try:
        latest = max((cd for tier in cells_by_tier for cd in
                      [Path(run["run_dir"]) / tier / f'cell_{c["name"]}'
                       for c in cells_by_tier[tier]]),
                     key=lambda p: p.stat().st_mtime)
        from datetime import datetime
        fresh = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    parts = []
    parts.append('<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">')
    _run_model = run.get("info", {}).get("model", "unknown")
    _run_instance = run.get("info", {}).get("instance", "unknown")
    parts.append(f'<title>GenAI Workload Characterization — {_run_model} / {_run_instance}</title>')
    parts.append('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>')
    parts.append(f'<style>{CSS}</style></head><body><div class="container">')
    parts.append(f'<h1>GenAI Workload Characterization — {_run_model}</h1>')
    parts.append(f'<p>Pipeline: BGE embed → Milvus HNSW → SeaweedFS → MongoDB → vLLM <strong>{_run_model}</strong> (GPU). '
                 f'Instance: <code>{_run_instance}</code>.</p>')

    # Meta card
    parts.append('<div class="meta-card">')
    parts.extend(tier_status_html)
    parts.append(f'<div class="meta-item"><div class="k">latest data</div><div class="v">{fresh}</div></div>')
    parts.append(f'<div class="meta-item"><div class="k">run dir</div><div class="v"><code>{Path(run["run_dir"]).name}</code></div></div>')
    # Derive sample counts per tier from the actual CSVs rather than hardcoding
    sample_counts = []
    for tier in ("tok64", "tok192", "tok320"):
        cells = cells_by_tier.get(tier, [])
        complete_cells = [c for c in cells if c["is_complete"] and not c["is_isolated"]]
        if complete_cells:
            ns = {c["stats"].get("n") for c in complete_cells if c["stats"]}
            ns.discard(None)
            n_str = ",".join(str(n) for n in sorted(ns)) if ns else "—"
        else:
            n_str = "—"
        sample_counts.append(f"{tier}={n_str}")
    parts.append(f'<div class="meta-item"><div class="k">queries per cell</div><div class="v">{" ".join(sample_counts)}</div></div>')
    parts.append('</div>')

    # Build data-quality notes dynamically
    _n64 = run.get("info", {}).get("count_64", 0)
    _dq_notes = []
    if _n64 and int(_n64) < 20:
        _dq_notes.append(f'<strong>Small sample (n={_n64}):</strong> latency values are profiling samples, '
                         f'not stable distributions. Do not interpret median/max differences across cells '
                         f'as statistically significant without more samples.')
    if run.get("info", {}).get("stream"):
        _all_cells = [c for t in cells_by_tier.values() for c in t if c["is_complete"] and c["stats"]]
        _tokens_missing = not any((c["stats"].get("n_out_mean") or 0) > 0 for c in _all_cells)
        if _tokens_missing:
            _dq_notes.append('<strong>Streaming run:</strong> <code>n_output_tokens</code> and '
                             '<code>backend_ms/tok</code> are <strong>not available</strong> — vLLM did not '
                             'return <code>usage.completion_tokens</code> in the stream. '
                             'TTFT, generation_ms, and inter-chunk_ms are real measured values. '
                             'True TPOT shows "—" where token count was unavailable.')
    _all_rag_cells = [c for t in cells_by_tier.values() for c in t
                      if c["mode"] == "rag" and c["is_complete"] and c["stats"]]
    _seaweed_zero = _all_rag_cells and not any((c["stats"].get("rag_seaweed_mean") or 0) > 0
                                               for c in _all_rag_cells)
    if _seaweed_zero:
        _dq_notes.append('<strong>SeaweedFS:</strong> <code>rag_seaweed_ms = 0</code> — chunks are served '
                         'inline from Milvus metadata, not fetched from SeaweedFS. The SeaweedFS hot path '
                         'is not exercised in this run.')
    parts.append('<div class="note">' +
                 '<br>'.join(f'⚠ {n}' for n in _dq_notes) +
                 '</div>')

    # ─── Methodology section ─────────────────────────────────────────────────
    info = run.get("info", {})
    stream_flag = info.get("stream")
    if stream_flag is not None:
        stream_label = "Yes (SSE — real TTFT/TPOT)" if stream_flag else "No (non-streaming)"
        parts.append(f'<p><strong>Streaming:</strong> {stream_label}</p>')
    s = run.get("stream", {})
    parts.append('<h2>How this benchmark was built &amp; what was measured</h2>')
    parts.append('<details class="detail"><summary>Show methodology &amp; how it was measured</summary>'
                 '<div class="detail-body">')
    parts.append('<p>The benchmark exercises the full LLM-service pipeline end-to-end and captures both '
                 'request-level latency telemetry and CPU hardware counters. Every cell below corresponds to '
                 'a specific (mode, input-bucket, output-token-budget) configuration. The orchestrator runs '
                 'a fresh batch of HTTP requests against the LLM service inside a Kubernetes pod while '
                 '<code>perf stat</code> attaches to that pod\'s PID 1 to measure CPU events for the duration.</p>')

    parts.append('<h3>The pipeline being measured</h3>')
    parts.append(f'<p>For each request: <strong>BGE encoder</strong> (BAAI bge-base-en-v1.5) computes a 768-dim '
                 f'embedding of the query → <strong>Milvus HNSW</strong> retrieves top-k similar chunks → '
                 f'<strong>SeaweedFS</strong> object store fetches the chunk JSON → context is formatted '
                 f'into the LLM prompt → <strong>vLLM</strong> backend runs <strong>{_run_model}</strong> '
                 f'on a separate GPU pod. In parallel, the <strong>semantic cache</strong> (BGE embed → '
                 f'Milvus similarity → MongoDB fetch) checks for a previously-cached response.</p>')

    parts.append('<h3>Pods — what each one is and does</h3>')
    parts.append('<table><thead><tr><th>Pod</th><th>Container image</th><th>Role</th></tr></thead><tbody>'
        '<tr><td><code>fastapi</code></td><td>llm-service/fastapi</td><td>'
        'Main service gateway and request orchestrator. Receives all client HTTP requests, '
        'runs the <strong>BGE-large embedding model on CPU</strong> (AVX-512 FP32 via PyTorch/ONNX) '
        'to encode queries, routes each request through the RAG and/or semantic-cache pipeline, '
        'assembles the augmented prompt, and proxies generation requests to vLLM. '
        'This is the only pod where SIMD FP32 activity is visible — it owns all CPU-side ML compute.'
        '</td></tr>'
        '<tr><td><code>vllm</code></td><td>vllm-openai</td><td>'
        'LLM inference server. Runs <strong>Qwen2.5 on GPU</strong> via the vLLM engine. '
        'Handles KV-cache management, continuous batching, and token generation. '
        'Exposes an OpenAI-compatible HTTP API. fastapi sends the assembled prompt here and streams tokens back. '
        'CPU activity in this pod is scheduling/control overhead — the actual matrix math is on the GPU.'
        '</td></tr>'
        '<tr><td><code>milvus</code></td><td>milvusdb/milvus</td><td>'
        'Vector database. Stores <strong>BGE embeddings</strong> for both the RAG document chunks and '
        'the semantic cache. On each request it performs an <strong>HNSW approximate nearest-neighbour search</strong> '
        'to find the top-k most similar chunks (RAG path) or the closest cached query (cache path). '
        'Dominant CPU activity: HNSW graph traversal — pointer-chasing through a high-dimensional graph, '
        'which is heavily Frontend-Bound (I-cache pressure from large graph structures).'
        '</td></tr>'
        '<tr><td><code>milvus_etcd</code></td><td>bitnami/etcd</td><td>'
        'Metadata store for Milvus. Holds cluster topology, collection schema, segment assignments, '
        'and distributed lock state. Milvus workers consult etcd on startup and on schema changes; '
        'during steady-state query serving etcd sees little traffic. '
        'CPU profile: highly Frontend-Bound — etcd is a Go binary with large code footprint and many small allocations.'
        '</td></tr>'
        '<tr><td><code>milvus_minio</code></td><td>minio/minio</td><td>'
        'Object storage backend for Milvus. Stores serialised segment data (raw vectors, indexes) '
        'that Milvus loads into memory on startup or segment promotion. '
        'During query serving, MinIO is mostly idle unless Milvus flushes or loads a new segment. '
        'When active: sequential large reads from local disk → memory, moderate DRAM bandwidth.'
        '</td></tr>'
        '<tr><td><code>mongodb</code></td><td>mongo</td><td>'
        'Document store for the <strong>semantic cache</strong>. When a query hits the cache, '
        'MongoDB is queried by the cache key (Milvus returns the embedding ID, MongoDB returns the '
        'pre-computed LLM response). On a cache miss it receives a write with the new response. '
        'CPU profile: moderate Frontend-Bound due to WiredTiger storage engine code paths; '
        'DRAM traffic proportional to working-set size vs available RAM.'
        '</td></tr>'
        '<tr><td><code>seaweed_master</code></td><td>chrislusf/seaweedfs</td><td>'
        'SeaweedFS master server. Manages volume assignments, file-to-volume mapping metadata, '
        'and replication topology. Handles namespace lookups when a file is read or written. '
        'Light CPU load during steady-state serving.'
        '</td></tr>'
        '<tr><td><code>seaweed_volume</code></td><td>chrislusf/seaweedfs</td><td>'
        'SeaweedFS volume server. Stores the <strong>actual document chunk files</strong> (ingested PDFs → JSON chunks). '
        'The RAG pipeline fetches chunk content from here after Milvus returns the chunk IDs.'
        '</td></tr>'
        '<tr><td><code>seaweed_filer</code></td><td>chrislusf/seaweedfs</td><td>'
        'SeaweedFS filer. Provides a POSIX-style filesystem layer over the volume servers, '
        'used for ingestion (PDF upload) and document management. '
        'Mostly idle during inference serving.'
        '</td></tr>'
        '</tbody></table>')

    parts.append('<h3>Modes (which paths run)</h3>')
    parts.append('<ul style="line-height:1.7">'
                 '<li><strong>rag</strong>: <code>bypass_rag=False, bypass_cache=True</code> — RAG runs, cache off, LLM always called. Measures full RAG pipeline.</li>'
                 '<li><strong>sc_a</strong>: <code>bypass_rag=True, bypass_cache=False</code> — RAG off, cache on. Isolated semantic cache (no RAG cost).</li>'
                 '<li><strong>sc_b</strong>: <code>bypass_rag=False, bypass_cache=False</code> — production flow: RAG runs first, then cache. On hit, LLM is skipped.</li>'
                 '<li><strong>llm_direct</strong>: <code>bypass_rag=True, bypass_cache=True</code> — no RAG, no cache, just HTTP forwarding to vLLM. Pure GPU baseline.</li>'
                 '<li><strong>bge_isolated</strong>, <strong>hnsw_isolated</strong>: micro-benchmarks of single components (50 queries, no LLM).</li>'
                 '</ul>')

    parts.append('<h3>Dimensions varied</h3>')
    parts.append('<ul style="line-height:1.7">'
                 '<li><strong>Token tier (output budget)</strong>: <code>tok64</code> / <code>tok192</code> / <code>tok320</code>. The same input queries are reused across tiers; only <code>--max-tokens</code> changes.</li>'
                 '<li><strong>Bucket (input length)</strong>: <code>short</code> ~30t, <code>medium</code> ~100t, <code>long</code> ~400t, <code>very_long</code> ~1600t.</li>'
                 f'<li><strong>Sample count per cell</strong>: {info.get("count_64","?")} at tok64, '
                 f'{info.get("count_192","?")} at tok192, {info.get("count_320","?")} at tok320 '
                 f'(from run_info.json — small n means p95/p99 are not meaningful statistics).</li>'
                 '</ul>')

    parts.append('<h3>The 6 perf passes per cell</h3>')
    parts.append('<p>Each cell runs the full request batch <strong>six times</strong>, with a different '
                 'set of perf events each pass (PMU has limited hardware counters, so we trade time for '
                 'counter breadth). Pass 2 is split into two sub-passes (2a/2b) to avoid '
                 'PMU multiplexing (more events than hardware counters, which scales counts '
                 'and reduces accuracy):</p>')
    parts.append('<ul style="line-height:1.7">'
                 '<li><strong>pass1</strong>: <code>cycles, instructions, task-clock, branch-misses, uops_issued.any, uops_retired.slots, context-switches, cpu-migrations</code> → IPC, branch-miss rate, average frequency, speculation waste, scheduling activity.</li>'
                 '<li><strong>pass2a</strong>: <code>L1-dcache-load-misses, l2_rqsts.miss, cache-misses, cache-references</code> — L1/L2/LLC miss counts (split from pass2b to avoid PMU multiplexing).</li>'
                 '<li><strong>pass2b</strong>: <code>cycle_activity.stalls_l3_miss, cycle_activity.stalls_total, iTLB-load-misses, L1-icache-load-misses, dTLB-load-misses</code> → stall fractions + TLB MPKI.</li>'
                 '<li><strong>pass3</strong>: <code>uncore_imc/cas_count_read/, uncore_imc/cas_count_write/</code> system-wide (<code>-a</code>) — real DRAM bandwidth (average over wall window; peak per interval on EKS with --interval-print). Reported as a single headline + STREAM sanity check, since node-wide IMC has no per-PID attribution.</li>'
                 '<li><strong>pass4</strong>: <code>cycles, exe_activity.bound_on_loads, l1d_pend_miss.pending/pending_cycles</code> — load-bound fraction + MLP. (Merged former 5a/5b; the L1/L2/L3-hit pyramid, store-bound and ports_util were dropped as redundant with TMA\'s Memory_Bound/Core_Bound decomposition.)</li>'
                 '</ul>')

    parts.append('<h3>TMA top-down + pass3 sweep</h3>')
    parts.append('<p>After the 4 main passes for each tier, the script runs:</p>'
                 '<ul style="line-height:1.7">'
                 '<li><strong>TMA</strong> (Top-down Microarchitecture Analysis): a separate 50-query pass per mode at <code>medium</code> bucket, captured with both Intel <code>pmu-tools/toplev.py -l1</code> and <code>perf stat topdown-*</code> events. Splits each pipeline slot into Retiring / FE-Bound / BE-Bound / Bad-Speculation buckets. Runs at all 3 tiers (tok64/tok192/tok320) with <code>|| warn</code> fallback — a TMA failure on one tier does not skip the others.</li>'
                 '<li><strong>pass3 sweep (tok64 only)</strong>: re-runs pass3+4 for every tok64 cell with cleaner statistical sampling. Per-request CSVs accumulate (timestamped sibling files, never overwritten); only the perf .txt at cell root is refreshed.</li>'
                 '<li><strong>STREAM calibration</strong>: 96-thread STREAM benchmark on the host before the run begins, giving the real Triad/Scale/Copy/Add ceilings to compare DRAM utilization against.</li>'
                 '<li><strong>perf-verify</strong>: a 100ms <code>sleep 0.1</code> with all our perf events attached, confirming each event encoding fires on this Sapphire Rapids host.</li>'
                 '</ul>')

    parts.append('<h3>Tools used</h3>')
    parts.append('<ul style="line-height:1.7">'
                 '<li><code>perf stat</code> (Linux <code>perf_event_open</code> API) for hardware-counter measurement.</li>'
                 '<li>Intel <a href="https://github.com/andikleen/pmu-tools">pmu-tools</a> <code>toplev.py</code> for TMA classification.</li>'
                 '<li><a href="https://www.cs.virginia.edu/stream/">STREAM</a> benchmark for DRAM ceiling calibration.</li>'
                 '<li><code>kubectl exec</code> to attach perf inside the running LLM-service pod (PID 1 = FastAPI orchestrator).</li>'
                 '<li><code>scripts/query_runner.py</code> — HTTP client with configurable client-side concurrency (<code>--concurrency N</code> fires N queries in parallel via a ThreadPoolExecutor). Captures per-request CSV with all sub-phase timings injected by the orchestrator. Streaming mode (<code>--stream</code>) captures real TTFT + TPOT via SSE.</li>'
                 '<li><code>scripts/run_benchmark.sh</code> — bash orchestrator that drives the 6-pass cycle per cell across all (mode, bucket, tier) combinations, plus per-tier TMA.</li>'
                 '<li><code>scripts/vllm_metrics_scraper.py</code> — polls the vLLM <code>/metrics</code> endpoint during pass 1 (queue depth, KV-cache %, preemptions, token throughput).</li>'
                 '<li><code>scripts/run_tma_extra.sh</code> — legacy manual TMA fallback (superseded: <code>run_tma()</code> now runs at every tier automatically).</li>'
                 '<li><code>scripts/generate_report.py</code> — this report. Parses every perf .txt and CSV, derives metrics, runs the integrity audit, renders HTML.</li>'
                 '</ul>')

    parts.append('<h3>Hardware &amp; software</h3>')
    parts.append('<table><thead><tr><th>property</th><th>value</th></tr></thead><tbody>')
    parts.append(f'<tr><td>Instance</td><td><code>{info.get("instance", "—")}</code> ({info.get("node", "—")})</td></tr>')
    parts.append(f'<tr><td>CPU</td><td>Intel Xeon Platinum 8488C / Sapphire Rapids, 96 hardware threads</td></tr>')
    parts.append(f'<tr><td>Base frequency (verified)</td><td>~3.6 GHz (from <code>calibration/perf_verify.txt</code>: 1395027 cycles in 0.39 ms)</td></tr>')
    parts.append(f'<tr><td>STREAM ceiling on this host</td><td>'
                 f'Triad {fmt_v(s.get("triad_gbs"), " GB/s")} · '
                 f'Scale {fmt_v(s.get("scale_gbs"), " GB/s")} · '
                 f'Copy {fmt_v(s.get("copy_gbs"), " GB/s")} · '
                 f'Add {fmt_v(s.get("add_gbs"), " GB/s")}</td></tr>')
    parts.append(f'<tr><td>Model</td><td>{info.get("model", "—")} (vLLM backend on separate GPU pod via Istio gateway)</td></tr>')
    parts.append(f'<tr><td>Embedder</td><td>BAAI bge-base-en-v1.5 (768-dim), running on the FastAPI pod CPU (AVX-512)</td></tr>')
    parts.append(f'<tr><td>Vector DB</td><td>Milvus HNSW (RAG collection: <code>rag_chunks_seaweed_v2</code>; SC collection: <code>semcache_direct_v2</code>)</td></tr>')
    parts.append(f'<tr><td>RAG content store</td><td>SeaweedFS object store '
                 f'({"forced on hot path" if str(info.get("seaweed_force","0")) == "1" else "default"})</td></tr>')
    parts.append(f'<tr><td>Run timestamp</td><td>{info.get("timestamp", "—")}</td></tr>')
    parts.append(f'<tr><td>Sample count per cell</td><td>tok64: {info.get("count_64", "—")} · tok192: {info.get("count_192", "—")} · tok320: {info.get("count_320", "—")}</td></tr>')
    parts.append(f'<tr><td>Per-pass warmup count (SC modes only)</td><td>{info.get("warmup", "—")}</td></tr>')
    parts.append(f'<tr><td>Pre-warmup count (per cell, outside perf window)</td><td>{info.get("prerun_warmup", "—")}</td></tr>')
    parts.append(f'<tr><td>Client-side concurrency</td><td>{info.get("concurrency", "—")} (queries fired in parallel via ThreadPoolExecutor; vLLM continuous batching absorbs them server-side)</td></tr>')
    parts.append(f'<tr><td>Streaming mode</td><td>{"on (SSE — real TTFT/TPOT for RAG+LLM_DIRECT cells; SC cells force non-stream so cache lookup runs)" if info.get("stream") else "off (TTFT proxied by frontend_overhead_ms)"}</td></tr>')
    parts.append('</tbody></table>')

    parts.append('</div></details>')  # end methodology collapsible

    # Per-tier sections
    for tier in ("tok64", "tok192", "tok320"):
        cells = cells_by_tier.get(tier, [])
        if not cells: continue
        complete = [c for c in cells if c["is_complete"]]
        partial = [c for c in cells if not c["is_complete"]]
        _donut_data = chart_data_cpu_gpu_donut(cells)
        _tma_data = run["tma"].get(tier)
        _bc_data = chart_data_bucket_components(cells)
        _bc_rag = _bc_data.get("rag", {})

        parts.append(f'<h2>{tier}</h2>')
        note_bits = [f'<strong>{len(complete)}/{len(cells)} cells complete</strong>']
        if partial:
            partial_desc = ", ".join(f'<code>{c["mode"]}_{c["bucket"]}</code> ({c["perf_passes"]}/5 passes)'
                                     for c in partial)
            note_bits.append(f'in-progress / partial: {partial_desc}')
        note_bits.append('Tables show <em>only fully-completed cells</em>.')
        parts.append(f'<div class="note">{". ".join(note_bits)}.</div>')

        # ── Primary charts visible by default ────────────────────────────────
        _gen_bucket_data = chart_data_generation_by_bucket(cells)
        _vllm_cpu_data = chart_data_vllm_cpu(cells)
        _primary = []
        if _donut_data:
            _primary.append(
                f'<div class="chart-card">'
                f'<p class="muted" style="margin:0 0 6px;font-size:11px;font-weight:600">CPU vs GPU time — where e2e latency goes</p>'
                f'<canvas id="chartDonut_{tier}" height="200"></canvas>'
                f'</div>'
            )
        for _bcm_mode in sorted(_bc_data.keys(), key=lambda x: MODE_ORDER.get(x, 99)):
            if len(_bc_data[_bcm_mode]) >= 2:
                _mode_label = _bcm_mode.replace('_', ' ')
                _primary.append(
                    f'<div class="chart-card">'
                    f'<p class="muted" style="margin:0 0 6px;font-size:11px;font-weight:600">'
                    f'Latency breakdown by input bucket — {_mode_label} (stacked by component)</p>'
                    f'<canvas id="chartBucketMain_{tier}_{_bcm_mode}" height="200"></canvas>'
                    f'</div>'
                )
        if _tma_data:
            _primary.append(
                f'<div class="chart-card">'
                f'<p class="muted" style="margin:0 0 6px;font-size:11px;font-weight:600">'
                f'TMA: CPU pipeline slot breakdown (% Retiring / FE-Bound / BE-Bound / Bad-Spec)</p>'
                f'<canvas id="chartTMA_{tier}" height="200"></canvas>'
                f'</div>'
            )
        if _gen_bucket_data:
            _gbd_streaming = any(v.get('is_streaming') for v in _gen_bucket_data.values())
            _gbd_title = (
                'Generation time vs input bucket — TTFT (real) + gen_ms (decode window) p50'
                if _gbd_streaming else
                'Latency vs input bucket — backend_ms (prefill+decode) + frontend_overhead_ms p50'
            )
            _primary.append(
                f'<div class="chart-card">'
                f'<p class="muted" style="margin:0 0 6px;font-size:11px;font-weight:600">'
                f'{_gbd_title}</p>'
                f'<canvas id="chartGenBucket_{tier}" height="200"></canvas>'
                f'</div>'
            )
        if _vllm_cpu_data.get("labels"):
            _primary.append(
                f'<div class="chart-card">'
                f'<p class="muted" style="margin:0 0 6px;font-size:11px;font-weight:600">'
                f'vLLM pod CPU — software counters only (Nitro hypervisor, no hardware PMU)</p>'
                f'<p class="muted" style="margin:0 0 6px;font-size:10px">'
                f'CPUs utilized = task-clock / wall-time. Near-zero confirms GPU does all compute; '
                f'CPU handles tokenisation, KV-cache bookkeeping and HTTP/SSE I/O only.</p>'
                f'<canvas id="chartVllmCpu_{tier}" height="200"></canvas>'
                f'</div>'
            )
        if _primary:
            parts.append('<div class="chart-grid">' + ''.join(_primary) + '</div>')

        # ── Everything else collapsed ─────────────────────────────────────────
        _detail: List[str] = []

        # Latency chart + numbers
        _detail.append('<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Latency</h3>')
        _detail.append(f'<div class="chart-card"><canvas id="chartLat_{tier}" height="80"></canvas></div>')
        _detail.append(collapsible("Show latency numbers", html_main_latency_table(cells)))

        # Pod/service breakdown per bucket (stacked bar)
        _detail.append(
            '<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Time per pod / service — mean e2e breakdown</h3>'
            '<p class="muted">Each bar = one bucket. Segments show where wall-clock time went: '
            'vLLM (GPU), BGE embed, Milvus, SeaweedFS, semantic-cache lookup, and untracked overhead.</p>'
            f'<div class="chart-card"><canvas id="chartPodBreak_{tier}" height="120"></canvas></div>'
        )

        # Pipeline / tail / payload charts
        _detail.append('<div class="chart-grid">')
        _detail.append(f'<div class="chart-card"><canvas id="chartPipeline_{tier}" height="180"></canvas></div>')
        _detail.append(f'<div class="chart-card"><canvas id="chartTail_{tier}" height="180"></canvas></div>')
        _detail.append(f'<div class="chart-card"><canvas id="chartPayload_{tier}" height="180"></canvas></div>')
        _detail.append('</div>')
        _detail.append(
            collapsible("Component breakdown numbers", html_pipeline_breakdown_table(cells)) +
            collapsible("Tail latency numbers", html_tail_latency_table(cells)) +
            collapsible("Payload &amp; retrieval numbers", html_request_payload_table(cells))
        )

        # Per-request timeline
        _tl = html_per_request_timeline(cells)
        if _tl:
            _detail.append(_tl)

        # Streaming latency
        _stream_cd = chart_data_streaming_latency(cells)
        if _stream_cd and _stream_cd["labels"]:
            _sl = json.dumps(_stream_cd["labels"])
            _st = json.dumps(_stream_cd["ttft_p50"])
            _sg = json.dumps(_stream_cd["gen_ms_p50"])
            _si = json.dumps(_stream_cd["ichunk_p50"])
            _sp = json.dumps(_stream_cd["tpot_p50"])
            _detail.append(
                '<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Streaming latency</h3>'
                f'<p class="muted"><strong>TTFT</strong>: time to first token. '
                f'<strong>gen_ms</strong>: first-to-last token window. '
                f'<strong>TPOT</strong>: gen_ms / (completion_tokens−1).</p>'
                f'<div class="chart-card"><canvas id="chartStream_{tier}" height="100"></canvas></div>'
                f'<script>new Chart(document.getElementById(\'chartStream_{tier}\'),{{'
                f'type:\'bar\','
                f'data:{{labels:{_sl},datasets:['
                f'{{label:\'TTFT p50 (ms)\',data:{_st},backgroundColor:\'rgba(88,166,255,0.7)\',borderColor:\'#58a6ff\',borderWidth:1}},'
                f'{{label:\'gen_ms p50 (ms)\',data:{_sg},backgroundColor:\'rgba(63,185,80,0.7)\',borderColor:\'#3fb950\',borderWidth:1}},'
                f'{{label:\'inter-chunk p50 (ms)\',data:{_si},backgroundColor:\'rgba(210,153,34,0.5)\',borderColor:\'#d29922\',borderWidth:1}},'
                f'{{label:\'TPOT p50 (ms)\',data:{_sp},backgroundColor:\'rgba(248,81,73,0.7)\',borderColor:\'#f85149\',borderWidth:1}}'
                f']}},'
                f'options:{{responsive:true,plugins:{{legend:{{position:\'top\'}},'
                f'tooltip:{{callbacks:{{label:ctx=>ctx.dataset.label+\': \'+ctx.parsed.y.toFixed(1)+\' ms\'}}}}}}}}'
                f'}})</script>'
            )

        # Isolation micro-benchmarks
        if any(c["is_isolated"] for c in cells):
            _detail.append('<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Component isolation micro-benchmarks (BGE embed / HNSW search)</h3>')
            _detail.append(f'<div class="chart-card"><canvas id="chartIsolated_{tier}" height="80"></canvas></div>')
            _detail.append(collapsible("Show isolation numbers", html_isolated_table(cells)))

        # GPU + vLLM metrics
        _vllm_tbl = html_vllm_metrics_table(cells)
        _detail.append(
            '<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">GPU &amp; vLLM engine</h3>'
            '<p class="muted"><strong>SM activity%</strong>: fraction of SMs active. '
            '<strong>HMMA tensor%</strong>: BF16/FP16 tensor core utilisation. '
            '<strong>DRAM activity%</strong>: HBM bandwidth utilisation.</p>'
            f'<div class="chart-card"><canvas id="chartGPU_{tier}" height="80"></canvas></div>'
        )
        _detail.append(collapsible("Show GPU metric numbers", html_gpu_table(cells)))
        if _vllm_tbl:
            _detail.append(
                '<p class="muted" style="margin-top:12px"><strong>vLLM Prometheus metrics</strong> — '
                'scheduler queue depth, KV-cache %, preemptions, tok/s.</p>'
                f'<div class="chart-card"><canvas id="chartVllm_{tier}" height="80"></canvas></div>'
            )
            _detail.append(collapsible("Show vLLM metric numbers", _vllm_tbl))

        # Request decomposition + all-mode bucket charts
        _detail.append(
            '<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Request time decomposition (all modes)</h3>'
            '<p class="muted">Median time per segment across all modes.</p>'
            f'<div class="chart-card"><canvas id="chartDecomp_{tier}" height="100"></canvas></div>'
        )
        if _bc_data:
            _bc_charts_list = []
            for _bc_mode in sorted(_bc_data.keys(), key=lambda x: MODE_ORDER.get(x, 99)):
                if len(_bc_data[_bc_mode]) >= 2:
                    _bc_charts_list.append(f'<div class="chart-card"><canvas id="chartBucket_{tier}_{_bc_mode}" height="180"></canvas></div>')
            if _bc_charts_list:
                _detail.append('<div class="chart-grid">' + ''.join(_bc_charts_list) + '</div>')

        # Hardware counters + per-pod
        _hw_inner = (
            f'<div class="chart-card"><canvas id="chartHW_{tier}" height="80"></canvas></div>'
            + collapsible("Show hardware-counter table", html_hw_table(cells))
        )
        cells_with_pod_data = [c for c in cells if c.get("per_pod_hw") and not c["is_isolated"]]
        if cells_with_pod_data:
            _hw_inner += '<p class="muted" style="margin-top:10px">ILP = uops_executed.core / cycles. MLP = l1d_pend_miss / pending_cycles. AMAT = avg memory access time (cycles).</p>'
            for c in cells_with_pod_data[:6]:
                _hw_inner += collapsible(f'Per-pod counters — {c["mode"]} / {c["bucket"]}',
                                         html_cross_pod_table(c))
        _detail.append('<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">Hardware counters &amp; cross-pod comparison</h3>')
        _detail.append(_hw_inner)

        # TMA narratives (per-pod)
        per_pod_tma = run.get("per_pod_tma", {})
        if tier in per_pod_tma and per_pod_tma[tier]:
            _tma_narr: List[str] = [
                '<p class="muted">Top-down Microarchitecture Analysis at level 2. '
                'Bar: green=Retiring, blue=FE-Bound, yellow=Bad-Spec, red=BE-Bound.</p>'
            ]
            for mode in ("rag", "llm_direct", "sc_a"):
                mode_tma = per_pod_tma[tier].get(mode, {})
                if not mode_tma:
                    continue
                medium_cells = [c for c in cells if c["mode"] == mode and c["bucket"] == "medium"]
                ctx_cell = medium_cells[0] if medium_cells else {"per_pod_hw": {}, "mode": mode}
                _tma_narr.append(collapsible(f'TMA per-pod interpretation — {mode} / medium',
                                             html_tma_narrative_section(ctx_cell, mode_tma)))
            _detail.append('<h3 style="color:var(--accent);font-size:15px;margin:16px 0 8px">TMA interpretation (per-pod, level-2)</h3>')
            _detail.append("".join(_tma_narr))

        parts.append(section_fold("Show all measurements", "".join(_detail)))

    # ── Global comparison sections ────────────────────────────────────────────
    primary = cells_by_tier.get("tok64", [])
    if primary:
        # Row: mode comparison + DRAM bandwidth side by side
        parts.append('<h2>Cross-mode comparison</h2>')
        parts.append('<div class="chart-grid-2">')
        parts.append('<div class="chart-card">'
                     '<p class="muted" style="margin:0 0 6px;font-size:11px">e2e median latency by mode/bucket</p>'
                     '<canvas id="chartModeLat" height="200"></canvas></div>')

        # DRAM BW card (second column of chart-grid-2 started above)
        s = run["stream"]
        _dram_cells = [c for c in primary if not c["is_isolated"] and c["is_complete"]
                       and c["hw"].get("dram_avg_total_gbs") is not None]
        _dram_card = '<div class="chart-card">'
        _dram_card += '<p class="muted" style="margin:0 0 6px;font-size:11px">DRAM bandwidth (cas_count_read/write)</p>'
        if _dram_cells:
            _peak_cell = max(_dram_cells, key=lambda c: c["hw"]["dram_avg_total_gbs"])
            _peak = _peak_cell["hw"]["dram_avg_total_gbs"]
            _triad = s.get("triad_gbs") if s else None
            _pct_str = (f' ({round(_peak/_triad*100,1)}% of STREAM Triad)' if _triad else '')
            _dram_card += (f'<p class="muted">Peak: <strong>{fmt_v(_peak, " GB/s")}</strong>{_pct_str}</p>'
                           f'<canvas id="chartBWdetail" height="160"></canvas>')
        else:
            _dram_card += '<p class="muted">No pass3 DRAM data in this run.</p>'
        _dram_card += '</div>'
        parts.append(_dram_card)
        parts.append('</div>')  # close chart-grid-2
        if _dram_cells:
            parts.append(collapsible("Show per-cell DRAM table", html_bandwidth_table(primary, run["stream"])))

        # Row: stall breakdown + SIMD mix side by side
        parts.append('<h2>CPU bottleneck analysis</h2>')
        parts.append('<div class="chart-grid">')
        parts.append('<div class="chart-card">'
                     '<p class="muted" style="margin:0 0 4px;font-size:11px">Stall breakdown</p>'
                     '<canvas id="chartStalls" height="160"></canvas></div>')
        parts.append('<div class="chart-card">'
                     '<p class="muted" style="margin:0 0 4px;font-size:11px">SIMD instruction mix (FP32)</p>'
                     '<canvas id="chartSIMD" height="160"></canvas></div>')
        parts.append('</div>')
        # Detect whether stalls_l3_miss appears non-zero anywhere in the data
        any_l3_stall = any(c["hw"].get("stalls_l3_pct") for c in primary
                          if c["is_complete"] and c["hw"].get("stalls_l3_pct") is not None)
        _stall_notes = ""
        if not any_l3_stall:
            _stall_notes += ('<div class="note"><strong>stalls_l3_miss = 0</strong> across all cells — '
                             'event encoding may differ on Sapphire Rapids. LLC-miss share column is unreliable.</div>')
        _stall_notes += ('<div class="note"><strong>Cross-pass note:</strong> '
                         'stall % uses cycles from pass1 as denominator; stalls measured in pass2/pass4. '
                         'SC cells with long perf windows may show &gt;100% — see hardware-counter table per tier.</div>')
        parts.append(section_fold("Stall &amp; SIMD details", _stall_notes + collapsible("Show stall-breakdown numbers", html_stalls_table(primary))))

    # TMA charts are now in per-tier sections above; keep a brief methodology note here.
    if run["tma"]:
        missing = [t for t in ("tok192", "tok320") if t not in run["tma"]]
        if missing:
            parts.append(f'<div class="note">Tiers without TMA data: <code>{", ".join(missing)}</code> — '
                         f'the <code>tma</code> cell was not run for these tiers.</div>')

    # Multi-token scaling
    parts.append('<h2>Multi-token scaling — medium bucket</h2>')
    parts.append('<p>e2e median latency at each token tier for the medium bucket. Only data points from cells '
                 'with full pass4 measurement are shown; tiers with no complete cells produce no point.</p>')
    parts.append('<div class="chart-card"><canvas id="chartScaling" height="80"></canvas></div>')

    # TMA path charts — one stacked bar per path, pods on Y-axis
    parts.append(html_tma_path_charts(run.get("per_pod_tma", {})))

    # Findings — qualitative only; concrete numbers are in the tables above so they
    # always reflect the actual data, never a stale narrative.
    parts.append('<h2>Read-the-tables guide</h2>')
    parts.append('<details class="detail"><summary>Show how to read the charts &amp; tables</summary>'
                 '<div class="detail-body">')
    parts.append('<p>Concrete numbers are kept in the tables (behind the "Show…" toggles) so this section '
                 'never drifts from the data. When reading the report, the relationships to look for:</p>')
    parts.append('<ul style="line-height:1.8;color:var(--text-dim)">')
    parts.append('<li><strong>How GPU-bound is the system?</strong> Compare avg DRAM GB/s vs STREAM Triad. '
                 'Compare e2e against backend-only model_backend_ms — the gap is the CPU/orchestrator cost.</li>')
    parts.append('<li><strong>Cache value:</strong> Compare sc_a / sc_b e2e against rag at the same bucket. '
                 'Speedup ratio = rag_e2e / sc_e2e.</li>')
    parts.append('<li><strong>RAG cost decomposition:</strong> rag_embed_ms + rag_milvus_ms + rag_seaweed_ms '
                 '(in the pipeline-component table) ≈ what RAG adds over llm_direct.</li>')
    parts.append('<li><strong>EOS effect:</strong> EOS% column shows what fraction of requests terminated before max_tokens. '
                 'When EOS% is high, raw e2e mean drops below p50; use TPOT (per-token) for fair cross-tier comparisons.</li>')
    parts.append('<li><strong>CPU character per mode:</strong> IPC and LLC MPKI in the hw counter table reveal '
                 'how much CPU work each mode actually does (vs just waiting on the GPU).</li>')
    parts.append('<li><strong>Bottleneck class:</strong> The TMA panels show whether the CPU is BE-bound (memory/execution stalls), '
                 'FE-bound (instruction supply starving), Retiring (productive), or Bad-Spec (wasted speculation).</li>')
    parts.append('</ul>')
    parts.append('</div></details>')  # end read-the-tables guide collapsible

    parts.append('<p class="muted">Generated by <code>scripts/generate_report.py</code></p>')

    # JS / Chart.js
    parts.append('<script>')
    parts.append('Chart.defaults.color = "#8b949e"; Chart.defaults.borderColor = "#30363d"; Chart.defaults.font.family = "system-ui";')

    # Per-tier table charts (one chart immediately under each table)
    for tier in ("tok64", "tok192", "tok320"):
        tcells = cells_by_tier.get(tier, [])
        if not tcells: continue

        # Latency bar — e2e mean only
        d = chart_data_latency(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartLat_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'e2e mean', data: {json.dumps(d['mean'])}, backgroundColor: '#3fb950' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'e2e ms'}}, beginAtZero:true }} }} }}
}});''')

        # Pod / service breakdown stacked bar
        pd = chart_data_pod_breakdown(tcells)
        if pd["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartPodBreak_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(pd['labels'])},
    datasets: [
      {{ label:'vLLM (GPU)',      data: {json.dumps(pd['vllm'])},    backgroundColor: '#f85149', stack:'s' }},
      {{ label:'BGE embed',       data: {json.dumps(pd['embed'])},   backgroundColor: '#58a6ff', stack:'s' }},
      {{ label:'Milvus',          data: {json.dumps(pd['milvus'])},  backgroundColor: '#3fb950', stack:'s' }},
      {{ label:'SeaweedFS',       data: {json.dumps(pd['seaweed'])}, backgroundColor: '#d29922', stack:'s' }},
      {{ label:'Semantic cache',  data: {json.dumps(pd['cache'])},   backgroundColor: '#bc8cff', stack:'s' }},
      {{ label:'Gateway/other',   data: {json.dumps(pd['other'])},   backgroundColor: '#6e7681', stack:'s' }},
    ]
  }},
  options: {{
    responsive:true,
    scales:{{
      x:{{ stacked:true }},
      y:{{ stacked:true, title:{{display:true,text:'mean ms'}}, beginAtZero:true }}
    }},
    plugins:{{
      tooltip:{{callbacks:{{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+' ms'}}}}
    }}
  }}
}});''')

        # Request payload stacked bar
        d = chart_data_payload(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartPayload_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'orig prompt tokens',     data: {json.dumps(d['orig'])},       backgroundColor: '#58a6ff', stack:'in' }},
      {{ label:'RAG-injected tokens',    data: {json.dumps(d['rag_inject'])}, backgroundColor: '#3fb950', stack:'in' }},
      {{ label:'output tokens (mean)',   data: {json.dumps(d['n_out'])},      backgroundColor: '#f78166', stack:'out' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ x:{{ stacked:true }}, y:{{ stacked:true, title:{{display:true,text:'tokens'}} }} }} }}
}});''')

        # Pipeline component stacked bar
        d = chart_data_pipeline(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartPipeline_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'rag_embed',    data: {json.dumps(d['rag_embed'])},    backgroundColor: '#58a6ff' }},
      {{ label:'rag_milvus',   data: {json.dumps(d['rag_milvus'])},   backgroundColor: '#3fb950' }},
      {{ label:'rag_seaweed',  data: {json.dumps(d['rag_seaweed'])},  backgroundColor: '#d29922' }},
      {{ label:'cache_lookup', data: {json.dumps(d['cache_lookup'])}, backgroundColor: '#bc8cff' }},
      {{ label:'backend (GPU)',data: {json.dumps(d['backend'])},      backgroundColor: '#f78166' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ x:{{ stacked:true }}, y:{{ stacked:true, title:{{display:true,text:'mean ms'}} }} }} }}
}});''')

        # Tail latency stacked bar (per-component p95)
        d = chart_data_tail_latency(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartTail_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'rag_embed max',    data: {json.dumps(d['rag_embed_p95'])},    backgroundColor: '#58a6ff' }},
      {{ label:'rag_milvus max',   data: {json.dumps(d['rag_milvus_p95'])},   backgroundColor: '#3fb950' }},
      {{ label:'rag_seaweed max',  data: {json.dumps(d['rag_seaweed_p95'])},  backgroundColor: '#d29922' }},
      {{ label:'cache_embed max',  data: {json.dumps(d['cache_embed_p95'])},  backgroundColor: '#bc8cff' }},
      {{ label:'cache_milvus max', data: {json.dumps(d['cache_milvus_p95'])}, backgroundColor: '#ffa657' }},
      {{ label:'cache_mongo max',  data: {json.dumps(d['cache_mongo_p95'])},  backgroundColor: '#f78166' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'max ms (= p95 for n=5)'}}, beginAtZero:true }} }} }}
}});''')

        # Hardware counter chart: IPC + freq overlaid
        d = chart_data_hw_counters(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartHW_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'IPC',          data: {json.dumps(d['ipc'])},      backgroundColor:'#58a6ff', yAxisID:'y' }},
      {{ label:'avg freq GHz', data: {json.dumps(d['freq'])},     backgroundColor:'#3fb950', yAxisID:'y' }},
      {{ label:'L1 MPKI',      data: {json.dumps(d['l1_mpki'])},  backgroundColor:'#d29922', yAxisID:'y2' }},
      {{ label:'LLC miss %',   data: {json.dumps(d['llc_miss'])}, backgroundColor:'#f85149', yAxisID:'y2' }},
    ]
  }},
  options: {{
    responsive:true,
    scales: {{
      y:  {{ position:'left',  title:{{display:true,text:'IPC / freq'}} }},
      y2: {{ position:'right', title:{{display:true,text:'MPKI / miss%'}}, grid:{{drawOnChartArea:false}} }}
    }}
  }}
}});''')

        # Isolated cells (only tok64)
        if any(c["is_isolated"] for c in tcells):
            d = chart_data_isolated(tcells)
            if d["labels"]:
                parts.append(f'''
new Chart(document.getElementById('chartIsolated_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'mean ms',   data: {json.dumps(d['mean'])},   backgroundColor:'#3fb950' }},
      {{ label:'median ms', data: {json.dumps(d['median'])}, backgroundColor:'#58a6ff' }},
      {{ label:'max ms',    data: {json.dumps(d['max'])},    backgroundColor:'#d29922' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'ms'}}, beginAtZero:true }} }} }}
}});''')

        # GPU utilisation chart (SM% / tensor% / HBM%)
        d = chart_data_gpu(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartGPU_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'SM activity %',   data: {json.dumps(d['sm'])},   backgroundColor:'#58a6ff' }},
      {{ label:'tensor (HMMA) %', data: {json.dumps(d['hmma'])}, backgroundColor:'#3fb950' }},
      {{ label:'HBM (DRAM) %',    data: {json.dumps(d['dram'])}, backgroundColor:'#d29922' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'% of peak'}}, beginAtZero:true, max:100 }} }} }}
}});''')

        # vLLM KV-cache utilisation chart (min/median/max %)
        d = chart_data_vllm(tcells)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartVllm_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'KV-cache % min',    data: {json.dumps(d['kv_min'])}, backgroundColor:'#3fb950' }},
      {{ label:'KV-cache % median', data: {json.dumps(d['kv_med'])}, backgroundColor:'#58a6ff' }},
      {{ label:'KV-cache % max',    data: {json.dumps(d['kv_max'])}, backgroundColor:'#f85149' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'KV-cache % used'}}, beginAtZero:true, max:100 }} }} }}
}});''')

        d_decomp = chart_data_decomposition(tcells)
        if d_decomp['labels']:
            parts.append(f'''
new Chart(document.getElementById('chartDecomp_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d_decomp['labels'])},
    datasets: [
      {{ label:'BGE embed (ms)',    data: {json.dumps(d_decomp['embed'])},     backgroundColor:'#388bfd', stack:'s' }},
      {{ label:'Milvus HNSW (ms)', data: {json.dumps(d_decomp['milvus'])},    backgroundColor:'#3fb950', stack:'s' }},
      {{ label:'SeaweedFS (ms)',   data: {json.dumps(d_decomp['seaweed'])},   backgroundColor:'#d29922', stack:'s' }},
      {{ label:'MongoDB (ms)',     data: {json.dumps(d_decomp['mongo'])},     backgroundColor:'#bc8cff', stack:'s' }},
      {{ label:'vLLM / GPU (ms)', data: {json.dumps(d_decomp['vllm'])},     backgroundColor:'#f85149', stack:'s' }},
      {{ label:'CPU overhead (ms)',data: {json.dumps(d_decomp['remainder'])},backgroundColor:'#6e7681', stack:'s' }},
    ]
  }},
  options: {{
    responsive:true,
    plugins:{{ legend:{{position:'right'}} }},
    scales:{{ x:{{ stacked:true }}, y:{{ stacked:true,
      title:{{display:true,text:'median ms (stacked)'}} }} }}
  }}
}});''')

        # Donut chart — CPU vs GPU time (2 segments)
        _donut_js = chart_data_cpu_gpu_donut(tcells)
        if _donut_js:
            _dn_cpu = _donut_js['cpu']
            _dn_gpu = _donut_js['gpu']
            parts.append(f'''
(function() {{
  var _donut_total = {_dn_cpu} + {_dn_gpu};
  new Chart(document.getElementById('chartDonut_{tier}'), {{
    type: 'doughnut',
    data: {{
      labels: ['CPU (RAG + cache + FastAPI)', 'GPU (vLLM generation)'],
      datasets: [{{
        data: [{_dn_cpu}, {_dn_gpu}],
        backgroundColor: ['#388bfd', '#f85149'],
        borderWidth: 2,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{position: 'right'}},
        tooltip: {{callbacks: {{label: function(ctx) {{
          return ctx.label + ': ' + ctx.parsed.toFixed(1) + ' ms (' + (_donut_total > 0 ? (ctx.parsed / _donut_total * 100).toFixed(1) : '0.0') + '%)';
        }}}}}}
      }}
    }}
  }});
}})();''')

        # chartBucketMain_{tier}_{mode} — one stacked bar per path, all in primary grid
        _bucket_label_order = ['short', 'medium', 'long', 'very_long']
        for _bcm_mode in sorted(_bc_data.keys(), key=lambda x: MODE_ORDER.get(x, 99)):
            _bc_main_data = _bc_data[_bcm_mode]
            _bcm_labels = [b for b in _bucket_label_order if b in _bc_main_data]
            if len(_bcm_labels) < 2:
                continue
            _bcm_embed     = [_bc_main_data[b]['embed']     for b in _bcm_labels]
            _bcm_milvus    = [_bc_main_data[b]['milvus']    for b in _bcm_labels]
            _bcm_seaweed   = [_bc_main_data[b]['seaweed']   for b in _bcm_labels]
            _bcm_mongo     = [_bc_main_data[b]['mongo']     for b in _bcm_labels]
            _bcm_vllm      = [_bc_main_data[b]['vllm']      for b in _bcm_labels]
            _bcm_remainder = [_bc_main_data[b]['remainder'] for b in _bcm_labels]
            parts.append(f'''
new Chart(document.getElementById('chartBucketMain_{tier}_{_bcm_mode}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(_bcm_labels)},
    datasets: [
      {{ label:'BGE Embed (ms)',       data: {json.dumps(_bcm_embed)},     backgroundColor:'#388bfd', stack:'s' }},
      {{ label:'Milvus HNSW (ms)',     data: {json.dumps(_bcm_milvus)},    backgroundColor:'#3fb950', stack:'s' }},
      {{ label:'SeaweedFS (ms)',       data: {json.dumps(_bcm_seaweed)},   backgroundColor:'#d29922', stack:'s' }},
      {{ label:'MongoDB (ms)',         data: {json.dumps(_bcm_mongo)},     backgroundColor:'#bc8cff', stack:'s' }},
      {{ label:'vLLM / GPU (ms)',      data: {json.dumps(_bcm_vllm)},      backgroundColor:'#f85149', stack:'s' }},
      {{ label:'FastAPI overhead (ms)',data: {json.dumps(_bcm_remainder)}, backgroundColor:'#6e7681', stack:'s' }},
    ]
  }},
  options: {{
    responsive:true,
    plugins:{{ legend:{{position:'right'}} }},
    scales:{{ x:{{ stacked:true }}, y:{{ stacked:true,
      title:{{display:true,text:'e2e latency (ms, stacked by component)'}} }} }}
  }}
}});''')

        # chartGenBucket_{tier} — generation_ms + TTFT p50 vs input bucket, line chart
        _gbd = chart_data_generation_by_bucket(tcells)
        if _gbd:
            _mode_colors = {'rag': ('#f85149', '#ff8080'), 'llm_direct': ('#388bfd', '#80b8ff')}
            _datasets = []
            for _gbd_mode, _gbd_vals in _gbd.items():
                _solid, _light = _mode_colors.get(_gbd_mode, ('#aaa', '#ccc'))
                _dl = _gbd_vals['decode_label']
                _tl = _gbd_vals['ttft_label']
                _datasets.append(
                    f'{{label:\'{_dl} — {_gbd_mode} (p50)\','
                    f'data:{json.dumps(_gbd_vals["gen_ms"])},'
                    f'borderColor:\'{_solid}\',backgroundColor:\'{_solid}33\','
                    f'tension:0.3,fill:false}}'
                )
                _datasets.append(
                    f'{{label:\'{_tl} — {_gbd_mode} (p50)\','
                    f'data:{json.dumps(_gbd_vals["ttft"])},'
                    f'borderColor:\'{_light}\',backgroundColor:\'{_light}33\','
                    f'borderDash:[5,4],tension:0.3,fill:false}}'
                )
            _gbd_labels = list(list(_gbd.values())[0]['buckets'])
            parts.append(f'''
new Chart(document.getElementById('chartGenBucket_{tier}'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(_gbd_labels)},
    datasets: [{",".join(_datasets)}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{position:'right'}} }},
    scales: {{
      y: {{ title: {{display:true, text:'latency (ms, p50)'}} }}
    }}
  }}
}});''')

        # vLLM CPU software counters chart (primary — always visible)
        _vc = chart_data_vllm_cpu(tcells)
        if _vc.get("labels"):
            parts.append(f'''
new Chart(document.getElementById('chartVllmCpu_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(_vc['labels'])},
    datasets: [
      {{ label:'CPUs utilized', data: {json.dumps(_vc['cpu_util'])},
         backgroundColor:'rgba(248,81,73,0.7)', yAxisID:'y' }},
      {{ label:'ctx-switches / req', data: {json.dumps(_vc['ctx_per_req'])},
         backgroundColor:'rgba(88,166,255,0.7)', yAxisID:'y2' }},
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{position:'top'}},
      tooltip: {{callbacks: {{
        label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(ctx.datasetIndex===0?4:1)
      }}}}
    }},
    scales: {{
      y:  {{ type:'linear', position:'left',  beginAtZero:true,
              title:{{display:true, text:'CPUs utilized (fraction)'}} }},
      y2: {{ type:'linear', position:'right', beginAtZero:true,
              grid:{{drawOnChartArea:false}},
              title:{{display:true, text:'ctx-switches / req'}} }}
    }}
  }}
}});''')

        # Bucket-component stacked bar charts (one per mode) — inside "all measurements"
        _bc_data = chart_data_bucket_components(tcells)
        for _bc_mode in sorted(_bc_data.keys(), key=lambda x: MODE_ORDER.get(x, 99)):
            _bc_buckets = _bc_data[_bc_mode]
            if len(_bc_buckets) < 2:
                continue
            _bc_labels  = [b for b in _bucket_label_order if b in _bc_buckets]
            _bc_embed     = [_bc_buckets[b]['embed']     for b in _bc_labels]
            _bc_milvus    = [_bc_buckets[b]['milvus']    for b in _bc_labels]
            _bc_seaweed   = [_bc_buckets[b]['seaweed']   for b in _bc_labels]
            _bc_mongo     = [_bc_buckets[b]['mongo']     for b in _bc_labels]
            _bc_vllm      = [_bc_buckets[b]['vllm']      for b in _bc_labels]
            _bc_remainder = [_bc_buckets[b]['remainder'] for b in _bc_labels]
            parts.append(f'''
new Chart(document.getElementById('chartBucket_{tier}_{_bc_mode}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(_bc_labels)},
    datasets: [
      {{ label:'BGE Embed (ms)',      data: {json.dumps(_bc_embed)},     backgroundColor:'#388bfd', stack:'s' }},
      {{ label:'Milvus HNSW (ms)',    data: {json.dumps(_bc_milvus)},    backgroundColor:'#3fb950', stack:'s' }},
      {{ label:'SeaweedFS (ms)',      data: {json.dumps(_bc_seaweed)},   backgroundColor:'#d29922', stack:'s' }},
      {{ label:'MongoDB (ms)',        data: {json.dumps(_bc_mongo)},     backgroundColor:'#bc8cff', stack:'s' }},
      {{ label:'vLLM / GPU (ms)',     data: {json.dumps(_bc_vllm)},      backgroundColor:'#f85149', stack:'s' }},
      {{ label:'FastAPI overhead (ms)',data: {json.dumps(_bc_remainder)},backgroundColor:'#6e7681', stack:'s' }},
    ]
  }},
  options: {{
    responsive:true,
    plugins:{{ legend:{{position:'right'}}, title:{{display:true,text:'{_bc_mode} — component breakdown by bucket'}} }},
    scales:{{ x:{{ stacked:true }}, y:{{ stacked:true,
      title:{{display:true,text:'median ms (stacked)'}} }} }}
  }}
}});''')

    # Cross-tier stalls + prefetch + bandwidth-detail charts
    if primary:
        d = chart_data_stalls(primary)
        if d["labels"]:
            parts.append(f'''
new Chart(document.getElementById('chartStalls'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'stalls (total) %',  data: {json.dumps(d['stalls_tot'])},  backgroundColor:'#f85149' }},
      {{ label:'load-bound %',      data: {json.dumps(d['mem_stall'])},   backgroundColor:'#d29922' }},
      {{ label:'spec waste %',      data: {json.dumps(d['spec_waste'])},  backgroundColor:'#58a6ff' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'% of cycles (cross-pass — see caveat)'}}, beginAtZero:true }} }} }}
}});''')

    if primary:
        # Mode comparison chart
        d = chart_data_e2e_modes(primary)
        parts.append(f'''
new Chart(document.getElementById('chartModeLat'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'rag',         data: {json.dumps(d['rag'])},        backgroundColor: '#58a6ff' }},
      {{ label:'sc_a',        data: {json.dumps(d['sc_a'])},       backgroundColor: '#bc8cff' }},
      {{ label:'sc_b',        data: {json.dumps(d['sc_b'])},       backgroundColor: '#ffa657' }},
      {{ label:'llm_direct',  data: {json.dumps(d['llm_direct'])}, backgroundColor: '#f78166' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'e2e median ms'}}, beginAtZero:true }} }} }}
}});''')

        # SIMD chart (FP32 width mix). DRAM peak/avg and L1/L2/L3 pyramid charts
        # were removed — DRAM is reported as a single headline, and the pyramid is
        # covered by TMA.
        d = chart_data_simd(primary)
        parts.append(f'''
new Chart(document.getElementById('chartSIMD'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'AVX-512 FP32%', data: {json.dumps(d['avx512'])}, backgroundColor: '#58a6ff' }},
      {{ label:'AVX-256 FP32%', data: {json.dumps(d['avx256'])}, backgroundColor: '#3fb950' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ y:{{ title:{{display:true,text:'% of FP'}}, beginAtZero:true }} }} }}
}});''')

        d = chart_data_dram(primary)
        if d['labels']:
            parts.append(f'''
new Chart(document.getElementById('chartBWdetail'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'avg read GB/s',   data: {json.dumps(d['avg_read'])},   backgroundColor:'#58a6ff', stack:'avg' }},
      {{ label:'avg write GB/s',  data: {json.dumps(d['avg_write'])},  backgroundColor:'#3fb950', stack:'avg' }},
    ]
  }},
  options: {{ responsive:true, scales:{{ x:{{ stacked:true }}, y:{{ stacked:true, title:{{display:true,text:'GB/s'}} }} }} }}
}});''')

    for tier in ("tok64", "tok192", "tok320"):
        tier_data = run["tma"].get(tier)
        if not tier_data: continue
        d = chart_data_tma_tier(tier_data)
        parts.append(f'''
new Chart(document.getElementById('chartTMA_{tier}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'Retiring',     data: {json.dumps(d['retiring'])}, backgroundColor: '#3fb950' }},
      {{ label:'FE-Bound',     data: {json.dumps(d['fe_bound'])}, backgroundColor: '#d29922' }},
      {{ label:'BE-Bound',     data: {json.dumps(d['be_bound'])}, backgroundColor: '#f85149' }},
      {{ label:'Bad-Spec',     data: {json.dumps(d['bad_spec'])}, backgroundColor: '#bc8cff' }},
    ]
  }},
  options: {{
    indexAxis: 'y',
    responsive:true,
    scales: {{
      x:{{ stacked:true, max:100, title:{{display:true,text:'% slots'}} }},
      y:{{ stacked:true }}
    }}
  }}
}});''')

    # Scaling chart
    d = chart_data_scaling(cells_by_tier)
    parts.append(f'''
new Chart(document.getElementById('chartScaling'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(d['labels'])},
    datasets: [
      {{ label:'rag',         data: {json.dumps(d['rag'])},         borderColor:'#58a6ff', backgroundColor:'#58a6ff44', tension:0.2 }},
      {{ label:'sc_a',        data: {json.dumps(d['sc_a'])},        borderColor:'#bc8cff', backgroundColor:'#bc8cff44', tension:0.2 }},
      {{ label:'sc_b',        data: {json.dumps(d['sc_b'])},        borderColor:'#ffa657', backgroundColor:'#ffa65744', tension:0.2 }},
      {{ label:'llm_direct',  data: {json.dumps(d['llm_direct'])},  borderColor:'#f78166', backgroundColor:'#f7816644', tension:0.2 }},
    ]
  }},
  options: {{ responsive:true, scales:{{
    x:{{ title:{{display:true,text:'max_tokens'}} }},
    y:{{ title:{{display:true,text:'e2e median ms'}}, beginAtZero:true }} }} }}
}});''')

    parts.append('</script></div></body></html>')
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",   help="Run directory (default: latest under benchmark_results/)")
    parser.add_argument("--out",   help="Output HTML path (default: benchmark_report.html at repo root)")
    parser.add_argument("--tiers", help="Comma-separated token tiers to include, e.g. tok64 or tok64,tok192")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.run:
        run_dir = Path(args.run)
    else:
        results = repo_root / "benchmark_results"
        runs = sorted(results.glob("run_*"), key=lambda p: p.stat().st_mtime)
        if not runs:
            print("No runs found", file=sys.stderr)
            sys.exit(1)
        run_dir = runs[-1]
    print(f"Loading run: {run_dir}")

    out_path = Path(args.out) if args.out else (repo_root / "benchmark_report.html")
    tiers = [t.strip() for t in args.tiers.split(",")] if args.tiers else None
    run = load_run(run_dir, tiers)
    html = render_html(run)
    out_path.write_text(html)
    print(f"Wrote: {out_path}  ({len(html):,} chars)")


if __name__ == "__main__":
    main()
