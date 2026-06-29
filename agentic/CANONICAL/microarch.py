"""Shared, verified-correct microarch derivations for the agentic CPU-characterization plots.
Single source of truth so the FP/DRAM/L2 fixes can't drift between plot scripts.

Corrections baked in (see project_perf_issues #7/#9 + the 2026-06 audit):
  - FLOPs: precision-specific lanes (128b = 4 SP / 2 DP, 256b = 8/4, 512b = 16/8) and ×2 for FMA.
    (Old bug: summed SP+DP then applied the DP lane count to both -> SP-packed halved.)
  - AVX%: packed / (scalar+packed) in ELEMENTS (FMA ×2 cancels in the ratio).
  - DRAM: cgroup-scoped L3-miss×64 (container-only), NOT node-wide IMC (3-26x inflated by OS/page-cache).
  - L2 TMA: 4 measured L2 leaves + 4 derived siblings by subtraction.
"""
import collections, os

def parse(path):
    """perf 'stat -o' aggregate file -> {event: count}. First numeric token = value, next token = event."""
    a = collections.Counter()
    if not os.path.exists(path):
        return a
    for line in open(path, errors="ignore"):
        p = line.split()
        if len(p) < 2:
            continue
        try:
            v = float(p[0].replace(",", ""))
        except ValueError:
            continue
        if p[1] not in a:
            a[p[1]] = v
    return a

def ipc(a):
    return a.get("instructions", 0) / (a.get("cycles", 0) or 1)

# ---- FP / SIMD (corrected) ----
_LANES = {  # (single_elems, double_elems) per FP instruction by width
    "scalar": (1, 1), "128b_packed": (4, 2), "256b_packed": (8, 4), "512b_packed": (16, 8),
}
def _elements(a):
    g = lambda k: a.get("fp_arith_inst_retired." + k, 0)
    scalar = g("scalar_single") * 1 + g("scalar_double") * 1
    packed = 0.0
    for w, (sl, dl) in _LANES.items():
        if w == "scalar":
            continue
        packed += g(w + "_single") * sl + g(w + "_double") * dl
    return scalar, packed

def flops(a, fma=True):
    """Total FLOPs (×2 for FMA by the standard peak convention)."""
    scalar, packed = _elements(a)
    return (scalar + packed) * (2 if fma else 1)

def avx_pct(a):
    """Vectorized share = packed elements / total elements (FMA cancels)."""
    scalar, packed = _elements(a)
    tot = scalar + packed
    return packed / tot * 100 if tot else 0.0

# ---- cache + DRAM ----
def cache_hits(a):
    l1 = a.get("mem_load_retired.l1_hit", 0); l2 = a.get("mem_load_retired.l2_hit", 0)
    l3 = a.get("mem_load_retired.l3_hit", 0); miss = a.get("mem_load_retired.l3_miss", 0)
    tot = (l1 + l2 + l3 + miss) or 1; ins = a.get("instructions", 0) or 1
    return dict(l1=l1/tot*100, l2=l2/tot*100, l3=l3/tot*100, miss=miss/tot*100, mpki=miss/(ins/1000))

def dram_gb_cgroup(cache_dict):
    """Container-only DRAM proxy = L3 read-misses × 64 B (cgroup-scoped, read-miss lower bound)."""
    return cache_dict.get("mem_load_retired.l3_miss", 0) * 64 / 1e9

def dram_gb_nodewide(imc_dict):
    """Node-wide IMC (uncore) — kept for reference only; inflated by OS/other procs."""
    return (imc_dict.get("uncore_cha/unc_cha_imc_reads_count.normal/", 0)
            + imc_dict.get("uncore_cha/unc_cha_imc_writes_count.full/", 0)) * 64 / 1e9

# ---- MLP / ILP ----
def mlp(a): return a.get("l1d_pend_miss.pending", 0) / (a.get("l1d_pend_miss.pending_cycles", 0) or 1)
def ilp(a): return a.get("uops_executed.thread", 0) / (a.get("cycles", 0) or 1)

# ---- TMA ----
def tma_l1(a):
    sl = a.get("slots", 0) or 1
    return {k: a.get("topdown-" + k, 0) / sl * 100 for k in ("retiring", "fe-bound", "bad-spec", "be-bound")}

def tma_l2(a):
    """4 measured L2 leaves + 4 derived siblings (fetch-bw, light-ops, core-bound, machine-clears)."""
    sl = a.get("slots", 0) or 1
    pc = lambda k: a.get("topdown-" + k, 0) / sl * 100
    ret, fe, bs, be = pc("retiring"), pc("fe-bound"), pc("bad-spec"), pc("be-bound")
    fl, hv, mb, bm = pc("fetch-lat"), pc("heavy-ops"), pc("mem-bound"), pc("br-mispredict")
    return dict(retiring=ret, fe_bound=fe, bad_spec=bs, be_bound=be,
                fetch_lat=fl, fetch_bw=max(fe-fl, 0), heavy_ops=hv, light_ops=max(ret-hv, 0),
                mem_bound=mb, core_bound=max(be-mb, 0), br_mispred=bm, machine_clears=max(bs-bm, 0))
