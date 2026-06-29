# Canonical perf event groups for the agentic CPU-characterization suite.
# Source this. Each group is ONE perf pass (separate run) -> NO multiplexing.
# Validated on Intel Xeon w5-3425 (Sapphire Rapids) == same uarch as EKS c7i.metal-24xl.
#
# FIXES baked in vs the old per-harness scripts (see project_validation_findings):
#  - FP group includes packed-DOUBLE (was blind -> falsely reported "0% AVX")
#  - IMC uses uncore_cha (cas_count does NOT exist on EKS c7i.metal); bytes = count*64
#  - MLP/ILP uses uops_executed.thread (SMT-correct), not .core (up to 2x inflated)
#  - cycles+instructions in EVERY group for cross-check / per-group IPC

# Fixed-counter TMA (own pass -> stays 100%, never multiplexed)
PG_TMA="cycles,instructions,slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound"
# Pre-Ice-Lake CPUs (Cascade Lake / g4dn.metal Xeon 8259CL) lack the 'slots' metric counter ->
# LEGACY TMA from classic events (SLOTS = 4 * cpu_clk_unhalted.thread; 4-wide retire).
PG_TMA_LEGACY="cycles,instructions,cpu_clk_unhalted.thread,uops_issued.any,uops_retired.retire_slots,idq_uops_not_delivered.core,int_misc.recovery_cycles"
# Pick the correct TMA group for the running CPU. Test must CONFIRM 'slots' produces a real
# count, not just that perf exited 0 — a mismatched perf wrapper can exit 0 while printing only
# a warning, and Sapphire Rapids has 'slots' but NOT the legacy uops_retired.retire_slots event,
# so guessing wrong yields all-zero passes. Uses perf_bin() (a verified-working binary) if the
# shared lib is sourced, else falls back to bare 'perf'.
tma_group() {
  local PERF
  if type perf_bin >/dev/null 2>&1; then PERF="$(perf_bin 2>/dev/null)" || PERF=perf; else PERF=perf; fi
  if "$PERF" stat -e slots -- awk 'BEGIN{for(i=0;i<5000000;i++)s+=i}' 2>&1 | grep -qiE '[0-9][0-9,.]*[[:space:]]+slots'; then
    printf '%s' "$PG_TMA"
  else
    printf '%s' "$PG_TMA_LEGACY"
  fi
}

# Cache hierarchy -> AMAT, LLC-MPKI  (4 GP, clean)
PG_CACHE="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"

# Floating point / SIMD -> FLOPs, vectorization share  (FULL set incl packed-double)
PG_FP="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"

# Memory-level + instruction-level parallelism (SMT-correct ILP)
PG_MLP="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"

# TMA Level-2 drill-down: L1 + the 4 measured L2 leaves (derive 4 siblings by subtraction:
# fetch-bw=fe-fetch-lat, light-ops=ret-heavy-ops, core-bound=be-mem-bound, machine-clears=bad-spec-br-mispred).
# All fixed-counter / PERF_METRICS -> no multiplexing, cgroup-scopeable. Replaces the toplev.py pass
# (pruned to 3-5 nodes / degenerate on short windows). SPR/Ice-Lake+ only (same 'slots' requirement as PG_TMA).
PG_TD2="cycles,instructions,slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound,topdown-fetch-lat,topdown-heavy-ops,topdown-mem-bound,topdown-br-mispredict"

# Uncore DRAM bandwidth (CHA; bytes = count*64). NOTE: uncore CANNOT be cgroup-scoped -> node-wide.
PG_IMC="cycles,instructions,uncore_cha/unc_cha_imc_reads_count.normal/,uncore_cha/unc_cha_imc_writes_count.full/"
# Fallback for hosts where CHA is absent but legacy IMC exists (NOT EKS c7i.metal):
PG_IMC_FALLBACK="cycles,instructions,uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"

# All core-PMU groups that must run as separate passes (toplev + imc handled specially)
PG_CORE_GROUPS="TMA CACHE FP MLP"
