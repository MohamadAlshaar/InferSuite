#!/usr/bin/env python3
# =============================================================================
# vllm_metrics_scraper.py — poll a vLLM Prometheus /metrics endpoint during a
# benchmark cell and summarise the engine-internal metrics.
#
# Polls until SIGTERM/SIGINT (sent by run_benchmark.sh at the end of Pass 1) or
# until --duration elapses, whichever comes first. Writes:
#   --out-csv      : raw long-format time series  (t_rel, metric, value)
#   --out-summary  : JSON summary.
#
# Aggregation matches the rest of the benchmark (low n → min / max / median, no
# percentiles):
#   GAUGES   (queue depth, KV-cache %) → {min, max, median, last, n}
#   COUNTERS (preemptions, tokens)     → {start, end, delta, rate_per_s}
#
# Stdlib only — no requests/prometheus_client dependency, so it runs anywhere
# python3 is present.
#
# Usage:
#   python3 vllm_metrics_scraper.py --url http://<vllm-pod-ip>:8200/metrics \
#       --interval 1 --out-csv vllm_metrics.csv --out-summary vllm_metrics_summary.json
# =============================================================================
import argparse
import json
import re
import signal
import statistics
import sys
import time
import urllib.request
from typing import Dict, List, Optional

# Gauges: instantaneous values → min/max/median over the poll window.
# (kv_cache_usage_perc is the modern name; gpu_cache_usage_perc the older one —
#  vLLM emits one or the other depending on version, so we track both.)
GAUGE_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
)

# Counters: cumulative → report (end - start) delta and per-second rate.
COUNTER_METRICS = (
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
)

# Matches:  vllm:metric_name{label="x",...} 12.34   (labels optional)
_LINE_RE = re.compile(
    r'^(vllm:[a-zA-Z_][a-zA-Z0-9_]*)(\{[^}]*\})?\s+([-+0-9.eE]+)\s*$'
)

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True


def scrape_once(url: str, timeout: float) -> Dict[str, float]:
    """Fetch /metrics once and return {metric_base_name: summed_value}.

    Label sets are summed per base name (e.g. per-model series collapse to one
    total), which is correct for counters and fine for these single-replica
    gauges.
    """
    out: Dict[str, float] = {}
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    for line in body.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        name, _labels, val = m.group(1), m.group(2), m.group(3)
        try:
            fval = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + fval
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="vLLM /metrics endpoint")
    ap.add_argument("--interval", type=float, default=1.0, help="poll interval seconds")
    ap.add_argument("--duration", type=float, default=1800.0,
                    help="safety cap seconds (normally stopped earlier by SIGTERM)")
    ap.add_argument("--timeout", type=float, default=2.0, help="per-request HTTP timeout")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    # series[metric] = list of (t_rel, value)
    series: Dict[str, List[tuple]] = {}
    t0 = time.monotonic()
    n_polls = 0
    n_errors = 0

    while not _STOP and (time.monotonic() - t0) < args.duration:
        tick = time.monotonic()
        t_rel = round(tick - t0, 3)
        try:
            sample = scrape_once(args.url, args.timeout)
            n_polls += 1
            for name, val in sample.items():
                series.setdefault(name, []).append((t_rel, val))
        except Exception:
            n_errors += 1
        # Keep a steady cadence regardless of fetch latency.
        sleep_for = args.interval - (time.monotonic() - tick)
        if sleep_for > 0:
            # Wake early if signalled.
            end = time.monotonic() + sleep_for
            while time.monotonic() < end and not _STOP:
                time.sleep(min(0.1, end - time.monotonic()))

    duration = round(time.monotonic() - t0, 3)

    # ── Raw CSV (long format) ───────────────────────────────────────────────
    with open(args.out_csv, "w") as f:
        f.write("t_rel,metric,value\n")
        for name in sorted(series):
            for t_rel, val in series[name]:
                f.write(f"{t_rel},{name},{val}\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary: Dict[str, object] = {
        "url": args.url,
        "interval_s": args.interval,
        "duration_s": duration,
        "n_polls": n_polls,
        "n_errors": n_errors,
        "gauges": {},
        "counters": {},
    }

    for name in GAUGE_METRICS:
        pts = series.get(name)
        if not pts:
            continue
        vals = [v for _, v in pts]
        summary["gauges"][name] = {
            "min": round(min(vals), 4),
            "median": round(statistics.median(vals), 4),
            "max": round(max(vals), 4),
            "last": round(vals[-1], 4),
            "n": len(vals),
        }

    for name in COUNTER_METRICS:
        pts = series.get(name)
        if not pts:
            continue
        vals = [v for _, v in pts]
        start, end = vals[0], vals[-1]
        delta = end - start
        rate = (delta / duration) if duration > 0 else None
        summary["counters"][name] = {
            "start": round(start, 4),
            "end": round(end, 4),
            "delta": round(delta, 4),
            "rate_per_s": round(rate, 4) if rate is not None else None,
        }

    with open(args.out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"vllm-metrics: {n_polls} polls, {n_errors} errors, {duration:.1f}s, "
          f"{len(summary['gauges'])} gauges, {len(summary['counters'])} counters "
          f"→ {args.out_summary}")
    # Non-fatal if nothing scraped — the report degrades gracefully.
    sys.exit(0)


if __name__ == "__main__":
    main()
