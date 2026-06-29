#!/usr/bin/env python3
"""GPU prefill/decode sweep runner.

Runs two controlled, single-stream (concurrency=1) sweeps directly against vLLM:

  PREFILL sweep   input = {128..8192},  output = 1   → isolates prefill  (TTFT)
  DECODE  sweep   input = 1,  output = {64..1024}     → isolates decode   (TPOT)

Output length is *forced* exact via ignore_eos + min_tokens, so each point does
identical work. GPU hardware (DCGM) and vLLM engine (/metrics) are scraped
passively during each measurement window. The CPU is deliberately NOT profiled
here — it is idle and already characterised by the main benchmark.

Each point: warmup is done once per sweep (not here); the first measured request
is discarded as a transient guard; the rest are recorded.

Usage:
  python3 run_sweeps.py [--config config.json] [--out-dir results] [--sweep prefill|decode|all]

Env overrides (take precedence over config.json):
  VLLM_BASE_URL, VLLM_METRICS_URL, DCGM_METRICS_URL, MODEL, MODEL_HF_REPO
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import llm_client
import prompts

HERE = Path(__file__).resolve().parent
SCRAPER = str(HERE / "prom_scraper.py")

# What we keep from each request, in CSV column order.
CSV_FIELDS = [
    "sweep", "point", "req_idx", "discarded",
    "target_input_tokens", "target_output_tokens",
    "prompt_tokens", "completion_tokens",
    "ttft_ms", "generation_ms", "tpot_ms", "e2e_ms",
    "n_chunks_with_content", "http_status", "error",
]


def log(msg: str) -> None:
    print(f"\033[1;34m[gpu-sweep]\033[0m {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"\033[1;32m  ✓\033[0m {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"\033[1;33m  ⚠\033[0m {msg}", flush=True)


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    # Env overrides — let the pod spec point at the live services.
    for env_key, cfg_key in [
        ("VLLM_BASE_URL", "vllm_base_url"),
        ("VLLM_METRICS_URL", "vllm_metrics_url"),
        ("DCGM_METRICS_URL", "dcgm_metrics_url"),
        ("MODEL", "model"),
    ]:
        if os.environ.get(env_key):
            cfg[cfg_key] = os.environ[env_key]
    return cfg


class Scrapers:
    """Manage the passive vLLM + DCGM pollers for one measurement point."""

    def __init__(self, cfg: dict, point_dir: Path):
        self.procs: List[subprocess.Popen] = []
        self.cfg = cfg
        self.point_dir = point_dir

    def _start(self, url: str, match: str, stem: str) -> None:
        if not url:
            return
        cmd = [
            sys.executable, SCRAPER,
            "--url", url, "--match", match,
            "--interval", str(self.cfg.get("scrape_interval_s", 0.5)),
            "--out-csv", str(self.point_dir / f"{stem}.csv"),
            "--out-summary", str(self.point_dir / f"{stem}_summary.json"),
        ]
        self.procs.append(subprocess.Popen(cmd))

    def __enter__(self) -> "Scrapers":
        self._start(self.cfg.get("vllm_metrics_url", ""), r"^vllm:", "vllm")
        self._start(self.cfg.get("dcgm_metrics_url", ""), r"^DCGM_FI_(PROF|DEV)_", "dcgm")
        if self.procs:
            time.sleep(1.0)  # let the first sample land before requests start
        return self

    def __exit__(self, *exc) -> None:
        for p in self.procs:
            p.send_signal(signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def run_point(
    cfg: dict, sweep: str, point_label: str,
    input_tokens: int, output_tokens: int, n: int,
    out_root: Path, writer: csv.DictWriter, model_repo: str,
    scrape: bool = True,
) -> None:
    """Run one (input, output) point: optional scrapers + n+1 requests (discard first)."""
    point_dir = out_root / sweep / point_label
    point_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompts.build_prompt(input_tokens, model_repo)
    discard_first = bool(cfg.get("discard_first", True))
    total = n + (1 if discard_first else 0)

    ctx = Scrapers(cfg, point_dir) if scrape else _Null()
    bad = 0
    with ctx:
        for i in range(total):
            discarded = discard_first and i == 0
            row = llm_client.send(
                cfg["vllm_base_url"], cfg["model"], prompt, output_tokens,
                force_exact=True, timeout_s=cfg.get("request_timeout_s", 600),
            )
            rec = {
                "sweep": sweep, "point": point_label, "req_idx": i,
                "discarded": int(discarded),
                "target_input_tokens": input_tokens,
                "target_output_tokens": output_tokens,
                **{k: row.get(k) for k in (
                    "prompt_tokens", "completion_tokens", "ttft_ms",
                    "generation_ms", "tpot_ms", "e2e_ms",
                    "n_chunks_with_content", "http_status", "error")},
            }
            writer.writerow(rec)
            # Per-point CSV too (self-contained alongside the scraper output).
            _append_point_csv(point_dir / "requests.csv", rec)
            if row.get("http_status") != 200:
                bad += 1
            elif not discarded and output_tokens > 1 and row.get("completion_tokens") != output_tokens:
                warn(f"{point_label} req {i}: got {row.get('completion_tokens')} tokens, expected {output_tokens}")
    tag = "warmup" if not scrape else f"n={n}"
    (ok if bad == 0 else warn)(f"{sweep}/{point_label} done ({tag}, {bad} errors)")


def _append_point_csv(path: Path, rec: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_sweep(cfg: dict, sweep: str, out_root: Path, writer: csv.DictWriter, model_repo: str) -> None:
    spec = cfg[f"{sweep}_sweep"]
    wu = spec["warmup"]
    log(f"{sweep.upper()} sweep — warmup ({wu['n']} reqs, in={wu['input_tokens']} out={wu['output_tokens']})")
    run_point(cfg, sweep, "_warmup", wu["input_tokens"], wu["output_tokens"], wu["n"],
              out_root, writer, model_repo, scrape=False)

    if sweep == "prefill":
        for it in spec["input_tokens"]:
            log(f"PREFILL point: input={it} tokens, output={spec['output_tokens']}")
            run_point(cfg, sweep, f"in{it}", it, spec["output_tokens"], spec["n"],
                      out_root, writer, model_repo, scrape=True)
    else:
        for ot in spec["output_tokens"]:
            log(f"DECODE point: input={spec['input_tokens']}, output={ot} tokens")
            run_point(cfg, sweep, f"out{ot}", spec["input_tokens"], ot, spec["n"],
                      out_root, writer, model_repo, scrape=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--sweep", choices=["prefill", "decode", "all"], default="all")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    model_repo = os.environ.get("MODEL_HF_REPO", "Qwen/Qwen2.5-14B-Instruct")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / f"run_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "run_info.json").write_text(json.dumps({
        "timestamp": stamp,
        "model": cfg["model"],
        "vllm_base_url": cfg["vllm_base_url"],
        "vllm_metrics_url": cfg.get("vllm_metrics_url", ""),
        "dcgm_metrics_url": cfg.get("dcgm_metrics_url", ""),
        "dcgm_enabled": bool(cfg.get("dcgm_metrics_url")),
        "concurrency": 1,
        "discard_first": cfg.get("discard_first", True),
        "prefill_sweep": cfg["prefill_sweep"],
        "decode_sweep": cfg["decode_sweep"],
    }, indent=2))

    if not cfg.get("dcgm_metrics_url"):
        warn("DCGM_METRICS_URL not set — GPU hardware metrics will be SKIPPED "
             "(latency + vLLM engine metrics still collected). Set it to scrape dcgm-exporter.")

    log(f"vLLM: {cfg['vllm_base_url']}  model: {cfg['model']}")
    log(f"Results → {out_root}")

    all_csv = out_root / "all_requests.csv"
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        sweeps = ["prefill", "decode"] if args.sweep == "all" else [args.sweep]
        for sweep in sweeps:
            run_sweep(cfg, sweep, out_root, writer, model_repo)

    ok(f"All sweeps complete → {out_root}")
    print(f"\nNext: python3 analyze.py {out_root}")


if __name__ == "__main__":
    main()
