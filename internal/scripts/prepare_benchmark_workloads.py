#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tenant_for_key(key: str) -> str:
    return "tenantA" if int(sha1_text(key), 16) % 2 == 0 else "tenantB"


def other_tenant(tenant: str) -> str:
    return "tenantB" if tenant == "tenantA" else "tenantA"


def unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        k = norm_ws(item).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(norm_ws(item))
    return out


def build_paraphrases(question: str) -> List[str]:
    q = norm_ws(question).strip()
    if not q:
        return []

    q_no_q = q[:-1] if q.endswith("?") else q
    lower_q = q_no_q[0].lower() + q_no_q[1:] if q_no_q else q_no_q

    variants = [
        q,
        f"Please answer briefly: {q}",
        f"In one sentence, {lower_q}?" if lower_q else q,
        f"Can you answer this concisely: {q}",
        f"Briefly, {lower_q}?" if lower_q else q,
    ]
    return unique_keep_order(variants)


def prepare_squad(
    squad_path: Path,
    split_name: str,
    max_rows: int | None,
    corpus_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    queries: List[Dict[str, Any]] = []
    semantic_groups: List[Dict[str, Any]] = []

    for i, row in enumerate(read_jsonl(squad_path)):
        if max_rows is not None and i >= max_rows:
            break

        sample_id = str(row.get("id", f"{split_name}_{i}"))
        title = norm_ws(str(row.get("title", "untitled")))
        context = norm_ws(str(row.get("context", "")))
        question = norm_ws(str(row.get("question", "")))
        answers_obj = row.get("answers", {}) or {}
        answers_text = answers_obj.get("text") or []
        expected_answer = norm_ws(str(answers_text[0])) if answers_text else ""

        if not context or not question:
            continue

        corpus_key = sha1_text(f"{title}\n{context}")
        tenant = tenant_for_key(corpus_key)

        if corpus_key not in corpus_by_id:
            corpus_by_id[corpus_key] = {
                "corpus_id": corpus_key,
                "dataset": split_name,
                "title": title,
                "source": f"SQuAD/{title}",
                "tenant": tenant,
                "text": context,
                "page": 1,
            }

        queries.append(
            {
                "dataset": split_name,
                "sample_id": sample_id,
                "title": title,
                "tenant": tenant,
                "wrong_tenant": other_tenant(tenant),
                "prompt": question,
                "expected_answer": expected_answer,
                "corpus_id": corpus_key,
                "expected_mode": "grounded_rag",
            }
        )

        semantic_groups.append(
            {
                "group_id": f"{split_name}:{sample_id}",
                "dataset": split_name,
                "mode": "rag",
                "tenant": tenant,
                "wrong_tenant": other_tenant(tenant),
                "expected_answer": expected_answer,
                "corpus_id": corpus_key,
                "variants": build_paraphrases(question),
            }
        )

    return queries, semantic_groups


def prepare_dolly(
    dolly_path: Path,
    max_rows: int | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    plain_rows: List[Dict[str, Any]] = []
    semantic_groups: List[Dict[str, Any]] = []

    for i, row in enumerate(read_jsonl(dolly_path)):
        if max_rows is not None and i >= max_rows:
            break

        instruction = norm_ws(str(row.get("instruction", "")))
        response = norm_ws(str(row.get("response", "")))
        context = norm_ws(str(row.get("context", "")))
        category = norm_ws(str(row.get("category", "unknown")))

        if not instruction:
            continue

        sample_id = f"dolly_{i}"
        tenant = tenant_for_key(sample_id)

        plain_rows.append(
            {
                "dataset": "dolly",
                "sample_id": sample_id,
                "tenant": tenant,
                "wrong_tenant": other_tenant(tenant),
                "prompt": instruction,
                "expected_answer": response,
                "category": category,
                "provided_context": context,
                "expected_mode": "plain_vllm",
            }
        )

        semantic_groups.append(
            {
                "group_id": f"dolly:{sample_id}",
                "dataset": "dolly",
                "mode": "plain",
                "tenant": tenant,
                "wrong_tenant": other_tenant(tenant),
                "expected_answer": response,
                "corpus_id": None,
                "variants": build_paraphrases(instruction),
            }
        )

    return plain_rows, semantic_groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad-train", required=True)
    parser.add_argument("--squad-val", required=True)
    parser.add_argument("--dolly", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-squad-train", type=int, default=1000)
    parser.add_argument("--max-squad-val", type=int, default=500)
    parser.add_argument("--max-dolly", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_by_id: Dict[str, Dict[str, Any]] = {}

    squad_train_queries, squad_train_sem = prepare_squad(
        Path(args.squad_train),
        "squad_train",
        args.max_squad_train,
        corpus_by_id,
    )
    squad_val_queries, squad_val_sem = prepare_squad(
        Path(args.squad_val),
        "squad_val",
        args.max_squad_val,
        corpus_by_id,
    )
    dolly_plain, dolly_sem = prepare_dolly(
        Path(args.dolly),
        args.max_dolly,
    )

    corpus_rows = list(corpus_by_id.values())
    semantic_groups = squad_train_sem + squad_val_sem + dolly_sem

    counts = {
        "squad_corpus": write_jsonl(out_dir / "squad_corpus.jsonl", corpus_rows),
        "squad_queries_train": write_jsonl(out_dir / "squad_queries_train.jsonl", squad_train_queries),
        "squad_queries_val": write_jsonl(out_dir / "squad_queries_val.jsonl", squad_val_queries),
        "dolly_plain": write_jsonl(out_dir / "dolly_plain.jsonl", dolly_plain),
        "semantic_groups": write_jsonl(out_dir / "semantic_groups.jsonl", semantic_groups),
    }

    metadata = {
        "counts": counts,
        "tenants": {
            "tenantA": sum(1 for row in corpus_rows if row["tenant"] == "tenantA"),
            "tenantB": sum(1 for row in corpus_rows if row["tenant"] == "tenantB"),
        },
        "notes": {
            "squad_corpus": "Deduplicated by (title, context).",
            "squad_queries_train": "Grounded queries for RAG benchmark.",
            "squad_queries_val": "Held-out grounded queries.",
            "dolly_plain": "Plain instruction-following workload.",
            "semantic_groups": "Prompt-variant groups for semantic-cache traffic.",
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
