#!/usr/bin/env python3
"""Generic Prometheus poller — used for BOTH vLLM /metrics and DCGM-exporter.

Polls a Prometheus text endpoint at a fixed cadence until SIGTERM/SIGINT (sent
by the runner at the end of a measurement point) or until --duration elapses.
Per-series (metric name + labels) time series are written as CSV, plus a JSON
summary (min/median/max/last per series).

Stdlib only — no external deps, so it runs anywhere python3 is present, exactly
like the existing scripts/vllm_metrics_scraper.py it generalises.

Usage:
  # vLLM engine metrics
  python3 prom_scraper.py --url http://vllm:8000/metrics --match '^vllm:' \
      --out-csv vllm.csv --out-summary vllm_summary.json

  # DCGM GPU metrics (per-GPU labels preserved)
  python3 prom_scraper.py --url http://dcgm-exporter:9400/metrics \
      --match '^DCGM_FI_(PROF|DEV)_' \
      --out-csv dcgm.csv --out-summary dcgm_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import statistics
import sys
import time
import urllib.request
from typing import Dict, List, Tuple

# metric{labels} value   (labels optional)
_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEnaN]+)\s*$")

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True


def scrape_once(url: str, match: "re.Pattern", timeout: float) -> Dict[str, float]:
    """Fetch /metrics once; return {full_series_key: value} for matching metrics.

    The series key keeps labels (e.g. gpu="0") so per-GPU DCGM series stay
    distinct. vLLM series with the same base name collapse naturally because we
    keep their (identical-enough) label sets as separate keys only when present.
    """
    out: Dict[str, float] = {}
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if not match.search(name):
            continue
        try:
            fval = float(val)
        except ValueError:
            continue
        out[name + labels] = fval
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--match", required=True, help="regex; only matching metric names are kept")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=3600.0, help="safety cap (normally SIGTERM'd earlier)")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    match = re.compile(args.match)

    series: Dict[str, List[Tuple[float, float]]] = {}
    t0 = time.monotonic()
    n_polls = n_errors = 0

    while not _STOP and (time.monotonic() - t0) < args.duration:
        tick = time.monotonic()
        t_rel = round(tick - t0, 3)
        try:
            for key, val in scrape_once(args.url, match, args.timeout).items():
                series.setdefault(key, []).append((t_rel, val))
            n_polls += 1
        except Exception:
            n_errors += 1
        sleep_for = args.interval - (time.monotonic() - tick)
        end = time.monotonic() + max(0.0, sleep_for)
        while time.monotonic() < end and not _STOP:
            time.sleep(min(0.1, end - time.monotonic()))

    duration = round(time.monotonic() - t0, 3)

    with open(args.out_csv, "w") as f:
        f.write("t_rel,series,value\n")
        for key in sorted(series):
            for t_rel, val in series[key]:
                f.write(f"{t_rel},{key},{val}\n")

    summary: Dict[str, object] = {
        "url": args.url,
        "match": args.match,
        "interval_s": args.interval,
        "duration_s": duration,
        "n_polls": n_polls,
        "n_errors": n_errors,
        "series": {},
    }
    for key, pts in series.items():
        vals = [v for _, v in pts if v == v]  # drop NaN
        if not vals:
            continue
        summary["series"][key] = {
            "min": round(min(vals), 5),
            "median": round(statistics.median(vals), 5),
            "max": round(max(vals), 5),
            "last": round(vals[-1], 5),
            "n": len(vals),
        }

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[prom_scraper] {args.match}: {n_polls} polls, {n_errors} errors, "
          f"{duration:.1f}s, {len(summary['series'])} series → {args.out_summary}")
    sys.exit(0)


if __name__ == "__main__":
    main()
