#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.results_file))
    if not rows:
        print("No rows found.")
        return 0

    route_counts = Counter()
    category_counts = Counter()
    exact_hits = 0
    semantic_hits = 0
    semantic_shadow_hits = 0
    rag_used = 0
    ok_rows = 0

    latencies: List[float] = []
    by_category_latency = defaultdict(list)

    for row in rows:
        req = row.get("request", {}) or {}
        resp = row.get("response", {}) or {}
        category = req.get("category", "unknown")
        category_counts[category] += 1

        if row.get("error") is not None:
            continue

        ok_rows += 1
        route = safe_get(resp, "_route", "route_taken", default="unknown")
        route_counts[route] += 1

        exact_hit = bool(safe_get(resp, "_cache", "exact_hit", default=False))
        semantic_hit = bool(safe_get(resp, "_cache", "semantic_hit", default=False))
        semantic_shadow_hit = bool(safe_get(resp, "_cache", "semantic", "shadow_hit", default=False))
        rag_used_flag = bool(safe_get(resp, "_rag", "used", default=False))
        e2e_ms = float(safe_get(resp, "_perf", "e2e_ms", default=0.0) or 0.0)

        exact_hits += int(exact_hit)
        semantic_hits += int(semantic_hit)
        semantic_shadow_hits += int(semantic_shadow_hit)
        rag_used += int(rag_used_flag)
        latencies.append(e2e_ms)
        by_category_latency[category].append(e2e_ms)

    print("\n=== Overall ===")
    print(f"total_rows: {len(rows)}")
    print(f"successful_rows: {ok_rows}")
    print(f"route_counts: {dict(route_counts)}")
    print(f"category_counts: {dict(category_counts)}")
    print(f"exact_hit_rate: {exact_hits / ok_rows:.4f}" if ok_rows else "exact_hit_rate: 0")
    print(f"semantic_hit_rate: {semantic_hits / ok_rows:.4f}" if ok_rows else "semantic_hit_rate: 0")
    print(f"semantic_shadow_hit_rate: {semantic_shadow_hits / ok_rows:.4f}" if ok_rows else "semantic_shadow_hit_rate: 0")
    print(f"rag_used_rate: {rag_used / ok_rows:.4f}" if ok_rows else "rag_used_rate: 0")
    if latencies:
        print(f"e2e_ms_avg: {sum(latencies) / len(latencies):.2f}")
        print(f"e2e_ms_p50: {percentile(latencies, 0.50):.2f}")
        print(f"e2e_ms_p95: {percentile(latencies, 0.95):.2f}")

    print("\n=== By category ===")
    for category, vals in sorted(by_category_latency.items()):
        print(
            f"{category}: "
            f"n={len(vals)} "
            f"avg={sum(vals)/len(vals):.2f} "
            f"p50={percentile(vals, 0.50):.2f} "
            f"p95={percentile(vals, 0.95):.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
