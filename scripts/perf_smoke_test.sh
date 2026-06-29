#!/usr/bin/env bash
# perf_smoke_test.sh — verify which perf counters are available on this node.
#
# Run on benchmark-runner (c7i.metal-24xl) or perf-agent (p5.4xlarge):
#   bash scripts/perf_smoke_test.sh
#
# Or remotely from the benchmark-runner pod:
#   kubectl exec -n llm-service perf-agent    -- bash /app/scripts/perf_smoke_test.sh
#   kubectl exec -n llm-service benchmark-runner -- bash /app/scripts/perf_smoke_test.sh
# =============================================================================
set -uo pipefail

RED=$'\033[1;31m'; GRN=$'\033[1;32m'; YLW=$'\033[1;33m'; RST=$'\033[0m'
_pass()    { printf "${GRN}  PASS${RST}  %s\n" "$*"; }
_partial() { printf "${YLW}  PART${RST}  %s\n" "$*"; }
_fail()    { printf "${RED}  FAIL${RST}  %s\n" "$*"; }

# ── Perf binary (mirrors run_benchmark.sh) ────────────────────────────────────
PERF="${PERF_HOST_BIN:-$(find /usr/lib/linux-tools* -maxdepth 2 -name perf -type f -executable 2>/dev/null | sort -V | tail -1 || true)}"
PERF="${PERF:-perf}"

# ── Root vs sudo ──────────────────────────────────────────────────────────────
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null || true
    sysctl -w kernel.nmi_watchdog=0          2>/dev/null || true
else
    SUDO="sudo"
    sudo sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null \
        || printf "${YLW}WARNING${RST}: could not set perf_event_paranoid\n"
fi

PASS_COUNT=0
FAIL_COUNT=0

# ── Event group tester ────────────────────────────────────────────────────────
# Runs perf stat against `sleep 0.3`, counts <not counted>/<not supported>.
# Shows which individual events failed when partial.
test_events() {
    local label="$1" events="$2"

    local out
    out=$(${SUDO} "${PERF}" stat -e "${events}" -- sleep 0.3 2>&1 || true)

    local total failed ok
    total=$(printf '%s' "${events}" | tr ',' '\n' | wc -l)
    failed=$(printf '%s' "${out}" | grep -cE '<not counted>|<not supported>' || true)
    ok=$(( total - failed ))

    if [ "${failed}" -eq 0 ]; then
        _pass "${label} (${ok}/${total})"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    elif [ "${ok}" -gt 0 ]; then
        _partial "${label} (${ok}/${total} — ${failed} unavailable)"
        printf '%s' "${out}" | grep -E '<not counted>|<not supported>' \
            | sed 's/^/          /' || true
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    else
        _fail "${label} (0/${total} — all unavailable)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
}

# ── Header ────────────────────────────────────────────────────────────────────
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  perf smoke test"
printf "  binary  : %s\n"  "${PERF}"
printf "  version : %s\n"  "$(${PERF} --version 2>/dev/null || echo unknown)"
printf "  kernel  : %s\n"  "$(uname -r)"
printf "  paranoid: %s\n"  "$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
printf "  user    : %s\n"  "$(id -un) (uid=$(id -u))"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# ── Pass 1: IPC / ILP / branch ────────────────────────────────────────────────
echo "Pass 1 — IPC / ILP / branch"
test_events "pass1" \
    "cycles,instructions,task-clock,branch-misses,branch-instructions,uops_issued.any,uops_retired.slots,uops_executed.core,context-switches,cpu-migrations"

# ── Pass 2a: L1/L2/LLC cache hierarchy ───────────────────────────────────────
echo
echo "Pass 2a — cache hierarchy (L1/L2/LLC)"
test_events "pass2a" \
    "cycles,L1-dcache-load-misses,l2_rqsts.miss,cache-misses,cache-references"

# ── Pass 2b: Stalls + TLB ────────────────────────────────────────────────────
echo
echo "Pass 2b — stalls + TLB"
test_events "pass2b" \
    "cycles,cycle_activity.stalls_l3_miss,cycle_activity.stalls_total,iTLB-load-misses,L1-icache-load-misses,dTLB-load-misses"

# ── Pass 3: SIMD / prefetch ───────────────────────────────────────────────────
echo
echo "Pass 3 — SIMD / HW prefetch"
test_events "pass3" \
    "cycles,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.512b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.scalar_double,l2_rqsts.hwpf_miss,l2_rqsts.all_hwpf"

# ── Pass 4: IMC DRAM bandwidth (node-wide uncore) ────────────────────────────
echo
echo "Pass 4 — IMC DRAM bandwidth (node-wide, -a)"
imc_out=$(${SUDO} "${PERF}" stat -a \
    -e "uncore_imc/cas_count_read/,uncore_imc/cas_count_write/" \
    -- sleep 0.5 2>&1 || true)
if printf '%s' "${imc_out}" | grep -qE '<not counted>|<not supported>|error|No such|Permission'; then
    _fail "pass4 IMC (uncore events unavailable — virtualized host?)"
    printf '%s' "${imc_out}" | grep -E '<not counted>|<not supported>|error|No such|Permission' \
        | head -3 | sed 's/^/          /' || true
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
else
    _pass "pass4 IMC (uncore_imc/cas_count_read+write/)"
    PASS_COUNT=$(( PASS_COUNT + 1 ))
fi

# ── Pass 5a: Memory load pyramid ─────────────────────────────────────────────
echo
echo "Pass 5a — memory load pyramid"
test_events "pass5a" \
    "cycles,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,exe_activity.bound_on_loads"

# ── Pass 5b: Store-bound + MLP ────────────────────────────────────────────────
echo
echo "Pass 5b — store-bound + MLP"
test_events "pass5b" \
    "cycles,exe_activity.bound_on_stores,exe_activity.1_ports_util,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"

# ── TMA: toplev.py ───────────────────────────────────────────────────────────
echo
echo "TMA — toplev.py (pmu-tools)"
if command -v toplev.py >/dev/null 2>&1; then
    tma_out=$(${SUDO} toplev.py --quiet -l1 -- sleep 0.3 2>&1 || true)
    if printf '%s' "${tma_out}" | grep -qE 'FE |BE |Bad |Retiring|%'; then
        _pass "toplev.py L1 TMA"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        _fail "toplev.py returned unexpected output"
        printf '%s' "${tma_out}" | head -5 | sed 's/^/          /' || true
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
else
    _fail "toplev.py not in PATH (pmu-tools not installed)"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  Result: ${GRN}%d pass${RST}  ${RED}%d fail/partial${RST}\n" \
    "${PASS_COUNT}" "${FAIL_COUNT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
