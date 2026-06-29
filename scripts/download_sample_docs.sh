#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_sample_docs.sh — download arXiv PDFs for RAG ingestion
#
# Attempts to extract arXiv IDs from vectara/open_ragbench.
# Falls back to a curated list of 30 representative ML/NLP papers if
# open_ragbench doesn't expose direct arXiv IDs.
#
# Idempotent: skips PDFs that already exist.
# Can be run standalone or called from setup.sh.
#
# Environment overrides:
#   DOCS_DIR        where to place PDFs (default: $REPO_ROOT/docs_RAG)
#   N_PAPERS        number of PDFs to download (default: 60)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

DOCS_DIR="${DOCS_DIR:-${REPO_ROOT}/docs_RAG}"
N_PAPERS="${N_PAPERS:-60}"

log() { printf '[download_sample_docs] %s\n' "$*"; }

main() {
  mkdir -p "${DOCS_DIR}"

  log "Downloading ${N_PAPERS} arXiv PDFs for RAG ingestion..."
  log "Target directory: ${DOCS_DIR}"
  log "Requires: pip install datasets"

  DOCS_DIR="${DOCS_DIR}" N_PAPERS="${N_PAPERS}" python3 - << 'PYEOF'
import os, urllib.request, time, sys
from pathlib import Path

docs_dir = Path(os.environ['DOCS_DIR'])
docs_dir.mkdir(parents=True, exist_ok=True)
n_papers = int(os.environ.get('N_PAPERS', 60))

# Curated list of representative ML/NLP/Systems arXiv papers for RAG testing.
# Covers: transformers, LLMs, RAG, embeddings, vector search, serving.
FALLBACK_PAPERS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3
    "2204.02311",  # PaLM
    "2302.13971",  # LLaMA
    "2303.08774",  # GPT-4 Technical Report
    "2304.01196",  # Vicuna
    "2305.10601",  # Orca
    "2307.09288",  # LLaMA 2
    "2309.01427",  # Mistral 7B
    "2310.06825",  # Mixtral MoE
    "2311.10122",  # Phi-1.5
    "2312.00752",  # Mamba
    "2401.02385",  # DeepSeek
    "2401.12945",  # InternLM 2
    "1907.11692",  # RoBERTa
    "1909.05858",  # ALBERT
    "2002.08909",  # REALM (retrieval-augmented)
    "2004.05150",  # RAG (Lewis et al.)
    "2010.11934",  # BEIR benchmark
    "2101.03961",  # Dense Passage Retrieval
    "2104.08691",  # ColBERT v2
    "2106.09685",  # BGE / bi-encoder retrieval
    "2109.01652",  # FAISS
    "2110.07178",  # HNSW efficiency analysis
    "2112.10752",  # WebGPT
    "2201.11903",  # Chain-of-Thought prompting
    "2203.02155",  # Self-Consistency
    "2205.01068",  # Flan-T5
    "2206.05802",  # Emergent Abilities of LLMs
]

paper_ids = []

# Try to get IDs from open_ragbench first
try:
    from datasets import load_dataset
    print('[download_sample_docs] Checking vectara/open_ragbench for arXiv IDs...', flush=True)
    ds = load_dataset('vectara/open_ragbench', split='train', trust_remote_code=True)
    sample = list(ds.take(3))
    if sample:
        print(f'[download_sample_docs] Fields: {list(sample[0].keys())}', flush=True)
    for item in ds:
        for field in ('arxiv_id', 'doc_id', 'source_id', 'paper_id'):
            val = str(item.get(field, '') or '').strip()
            if val and '.' in val and 5 <= len(val) <= 15:
                paper_ids.append(val)
        if len(paper_ids) >= n_papers:
            break
    if paper_ids:
        print(f'[download_sample_docs] Found {len(paper_ids)} arXiv IDs in open_ragbench', flush=True)
except Exception as e:
    print(f'[download_sample_docs] open_ragbench exploration: {e}', flush=True)

# Fall back to curated list if no IDs found in dataset
if not paper_ids:
    print('[download_sample_docs] Using curated representative paper list', flush=True)
    paper_ids = FALLBACK_PAPERS[:n_papers]

paper_ids = list(dict.fromkeys(paper_ids))[:n_papers]  # deduplicate, limit
print(f'[download_sample_docs] Downloading {len(paper_ids)} papers...', flush=True)

done = 0
skipped = 0
failed = 0
for pid in paper_ids:
    fname = pid.replace('/', '_') + '.pdf'
    out = docs_dir / fname
    if out.exists() and out.stat().st_size > 10_000:
        skipped += 1
        continue
    url = f'https://arxiv.org/pdf/{pid}.pdf'
    try:
        urllib.request.urlretrieve(url, out)
        if out.stat().st_size < 1000:
            out.unlink()
            failed += 1
            print(f'  skip {pid}: response too small')
        else:
            done += 1
            time.sleep(0.5)
            if done % 10 == 0:
                print(f'  {done}/{len(paper_ids)} downloaded', flush=True)
    except Exception as e:
        failed += 1
        print(f'  skip {pid}: {e}')

count = len(list(docs_dir.glob('*.pdf')))
print(f'[download_sample_docs] Done — {count} PDFs in {docs_dir} '
      f'({done} downloaded, {skipped} already present, {failed} failed)')
PYEOF
}

main "$@"
