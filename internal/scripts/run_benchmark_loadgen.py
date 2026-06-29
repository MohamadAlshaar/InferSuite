#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_line(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower() if gold else False


def contains_expected(pred: str, gold: str) -> bool:
    return gold.strip().lower() in pred.strip().lower() if gold else False


def get_answer_text(resp: Dict[str, Any]) -> str:
    try:
        return str(resp["choices"][0]["message"]["content"])
    except Exception:
        return ""


def post_chat(
    base_url: str,
    tenant: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-Tenant-Id": tenant},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


class WorkloadGenerator:
    def __init__(
        self,
        *,
        squad_train: List[Dict[str, Any]],
        squad_val: List[Dict[str, Any]],
        dolly_plain: List[Dict[str, Any]],
        semantic_groups: List[Dict[str, Any]],
        rng: random.Random,
    ):
        self.squad_train = squad_train
        self.squad_val = squad_val
        self.dolly_plain = dolly_plain
        self.semantic_groups = semantic_groups
        self.rng = rng

        self.history: List[Dict[str, Any]] = []
        self.semantic_seen: Dict[str, int] = {}

    def choose_category(self, weights: Dict[str, float]) -> str:
        names = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(names, weights=probs, k=1)[0]

    def build_request(self, category: str) -> Dict[str, Any]:
        if category == "grounded_rag":
            item = self.rng.choice(self.squad_train)
            return {
                "category": category,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "tenant": item["tenant"],
                "prompt": item["prompt"],
                "expected_answer": item["expected_answer"],
                "meta": item,
            }

        if category == "grounded_rag_val":
            item = self.rng.choice(self.squad_val)
            return {
                "category": category,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "tenant": item["tenant"],
                "prompt": item["prompt"],
                "expected_answer": item["expected_answer"],
                "meta": item,
            }

        if category == "plain_dolly":
            item = self.rng.choice(self.dolly_plain)
            return {
                "category": category,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "tenant": item["tenant"],
                "prompt": item["prompt"],
                "expected_answer": item["expected_answer"],
                "meta": item,
            }

        if category == "cross_tenant":
            item = self.rng.choice(self.squad_train)
            return {
                "category": category,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "tenant": item["wrong_tenant"],
                "prompt": item["prompt"],
                "expected_answer": item["expected_answer"],
                "meta": item,
            }

        if category == "exact_repeat":
            if not self.history:
                return self.build_request("grounded_rag")
            item = self.rng.choice(self.history)
            return {
                "category": category,
                "dataset": item["dataset"],
                "sample_id": item["sample_id"],
                "tenant": item["tenant"],
                "prompt": item["prompt"],
                "expected_answer": item.get("expected_answer", ""),
                "meta": item.get("meta", {}),
            }

        if category == "semantic_variant":
            if not self.semantic_groups:
                return self.build_request("plain_dolly")

            reusable = [g for g in self.semantic_groups if self.semantic_seen.get(g["group_id"], 0) > 0]
            if reusable and self.rng.random() < 0.7:
                group = self.rng.choice(reusable)
            else:
                group = self.rng.choice(self.semantic_groups)

            seen_count = self.semantic_seen.get(group["group_id"], 0)
            variants = group["variants"] or []
            if not variants:
                return self.build_request("plain_dolly")

            if seen_count <= 0:
                prompt = variants[0]
            else:
                prompt = self.rng.choice(variants[1:] or variants)

            self.semantic_seen[group["group_id"]] = seen_count + 1
            return {
                "category": category,
                "dataset": group["dataset"],
                "sample_id": group["group_id"],
                "tenant": group["tenant"],
                "prompt": prompt,
                "expected_answer": group.get("expected_answer", ""),
                "meta": group,
            }

        return self.build_request("plain_dolly")

    def remember(self, req: Dict[str, Any]) -> None:
        self.history.append(req)


def summarize_live(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "no results yet"
    last = results[-1]
    route = ((last.get("response") or {}).get("_route") or {}).get("route_taken")
    e2e = (((last.get("response") or {}).get("_perf") or {}).get("e2e_ms")) or 0.0
    avg = statistics.fmean(
        [(((r.get("response") or {}).get("_perf") or {}).get("e2e_ms")) or 0.0 for r in results]
    )
    return f"n={len(results)} last_route={route} last_e2e_ms={e2e:.2f} avg_e2e_ms={avg:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen2.5-0.5b")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--results-file", required=True)

    parser.add_argument("--w-grounded-rag", type=float, default=30.0)
    parser.add_argument("--w-grounded-rag-val", type=float, default=10.0)
    parser.add_argument("--w-exact-repeat", type=float, default=25.0)
    parser.add_argument("--w-semantic-variant", type=float, default=20.0)
    parser.add_argument("--w-plain-dolly", type=float, default=10.0)
    parser.add_argument("--w-cross-tenant", type=float, default=5.0)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    squad_train = read_jsonl(workdir / "squad_queries_train.jsonl")
    squad_val = read_jsonl(workdir / "squad_queries_val.jsonl")
    dolly_plain = read_jsonl(workdir / "dolly_plain.jsonl")
    semantic_groups = read_jsonl(workdir / "semantic_groups.jsonl")

    rng = random.Random(args.seed)
    gen = WorkloadGenerator(
        squad_train=squad_train,
        squad_val=squad_val,
        dolly_plain=dolly_plain,
        semantic_groups=semantic_groups,
        rng=rng,
    )

    weights = {
        "grounded_rag": args.w_grounded_rag,
        "grounded_rag_val": args.w_grounded_rag_val,
        "exact_repeat": args.w_exact_repeat,
        "semantic_variant": args.w_semantic_variant,
        "plain_dolly": args.w_plain_dolly,
        "cross_tenant": args.w_cross_tenant,
    }

    results: List[Dict[str, Any]] = []
    results_file = Path(args.results_file)
    if results_file.exists():
        results_file.unlink()

    for i in range(args.num_requests):
        category = gen.choose_category(weights)
        req = gen.build_request(category)

        started = time.time()
        try:
            response = post_chat(
                base_url=args.base_url,
                tenant=req["tenant"],
                model=args.model,
                prompt=req["prompt"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            error = None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            response = None
            error = {"type": "HTTPError", "status": e.code, "body": raw}
        except Exception as e:
            response = None
            error = {"type": type(e).__name__, "message": str(e)}

        ended = time.time()
        model_answer = get_answer_text(response) if isinstance(response, dict) else ""
        expected = req.get("expected_answer", "")

        row = {
            "request_index": i,
            "started_at": started,
            "ended_at": ended,
            "wall_ms": (ended - started) * 1000.0,
            "request": {
                "category": req["category"],
                "dataset": req["dataset"],
                "sample_id": req["sample_id"],
                "tenant": req["tenant"],
                "prompt": req["prompt"],
                "expected_answer": expected,
            },
            "response": response,
            "error": error,
            "evaluation": {
                "exact_match": exact_match(model_answer, expected),
                "contains_expected": contains_expected(model_answer, expected),
                "answer_text": model_answer,
            },
        }

        write_jsonl_line(results_file, row)
        results.append(row)

        if error is None:
            gen.remember(req)

        print(f"[{i + 1}/{args.num_requests}] {summarize_live(results)}")

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"Saved results to {results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
