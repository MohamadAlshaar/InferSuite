#!/usr/bin/env bash
# measure.sh — one command for every measurement campaign in this repo.
#
# Each subcommand is a thin, documented wrapper over a proven campaign kit (it sets the
# right data root / env and calls the kit's own staged runner). Nothing is reimplemented
# here; this is the single entry point so you don't have to remember four script locations.
#
#   ./measure.sh <campaign> <stage> [args]
#
# CAMPAIGNS
#   agents-swe   SWE-agent x GLM-5.2, long-horizon (SWE_clean: django/sympy/babel/fmt)
#   agents-oc    OpenClaw x GLM-5.2, long-horizon (OC_clean: jigsaw/pdf/web/scp)
#   service      local k3s service, isolated per-tier CPU/TMA campaign
#   plots        regenerate every figure set from banked data (no capture, no spend)
#   validate     run every validator over banked data
#
# STAGES (agents-* and service share these; run in this order the first time)
#   preflight    fail-fast environment checks (no spend, no state change)
#   dryrun       counter-group multiplexing gate
#   smoke        one short episode end-to-end (agents only)
#   campaign     the real capture (honors the per-campaign env below)
#   validate     3-layer validator over the campaign's data
#
# EXAMPLES
#   ./measure.sh agents-swe preflight
#   SWE_INSTANCES="django__django-16560" SWE_DRAIN_S=5400 ./measure.sh agents-swe campaign
#   OC_TASKS="jigsaw-med" ./measure.sh agents-oc campaign
#   ./measure.sh service campaign
#   ./measure.sh plots            # all figure sets
#   ./measure.sh plots agents-swe # one set
#
# Per-campaign env (defaults are the certified values; override on the command line):
#   agents-swe : SWE_INSTANCES SWE_SUBSET(verified) SWE_TEMP(0.6) SWE_DRAIN_S LOOP_GUARD_N(12) REPEATS(1)
#   agents-oc  : OC_TASKS OC_DRAIN_S LOOP_GUARD_N(12) REPEATS(1) OC_WATCHER(lineage)
#   service    : TIERS REC_SEC STAT_SEC
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLM="$REPO/local_agents/scripts/glm"
SVC="$REPO/local_service/scripts/iso"
PY=python3

usage(){ sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 1
CAMP="$1"; STAGE="${2:-}"; shift $(( $# >= 2 ? 2 : 1 ))

case "$CAMP" in
  agents-swe)
    : "${SWE_TEMP:=0.6}" "${LOOP_GUARD_N:=12}" "${REPEATS:=1}" "${WINSEC:=5}"
    export SWE_TEMP LOOP_GUARD_N REPEATS SWE_SUBSET SWE_INSTANCES SWE_DRAIN_S WINSEC
    export DATA_ROOT="${DATA_ROOT:-$REPO/local_agents/SWE_clean/data}"
    [ -n "$STAGE" ] || { echo "need a stage (preflight|dryrun|smoke|campaign|validate)"; exit 1; }
    [ "$STAGE" = campaign ] && set -- swe "$@"
    exec "$GLM/run_glm_campaign.sh" "$STAGE" "$@" ;;

  agents-oc)
    : "${LOOP_GUARD_N:=12}" "${REPEATS:=1}" "${OC_WATCHER:=lineage}" "${WINSEC:=5}"
    export LOOP_GUARD_N REPEATS OC_WATCHER OC_TASKS OC_DRAIN_S WINSEC
    export DATA_ROOT="${DATA_ROOT:-$REPO/local_agents/OC_clean/data}"
    [ -n "$STAGE" ] || { echo "need a stage (preflight|dryrun|smoke|campaign|validate)"; exit 1; }
    [ "$STAGE" = campaign ] && set -- oc "$@"
    exec "$GLM/run_glm_campaign.sh" "$STAGE" "$@" ;;

  service)
    [ -n "$STAGE" ] || { echo "need a stage (preflight|dryrun|campaign|all)"; exit 1; }
    exec "$SVC/run_service_campaign.sh" "$STAGE" "$@" ;;

  plots)
    which="${STAGE:-all}"
    swe(){ echo "[plots] SWE_clean"; env PLOT_SPEC="$REPO/local_agents/SWE_clean/plot_spec.json" $PY "$GLM/plot_glm_results.py"
           env PLOT_SPEC="$REPO/local_agents/SWE_clean/plot_spec.json" $PY "$GLM/plot_call_structure.py"
           env PLOT_SPEC="$REPO/local_agents/SWE_clean/plot_spec.json" $PY "$GLM/plot_internal_tools.py"
           env PLOT_SPEC="$REPO/local_agents/SWE_clean/plot_spec.json" $PY "$GLM/plot_calls_vs_bursts.py"; }
    oc(){  echo "[plots] OC_clean";  env PLOT_SPEC="$REPO/local_agents/OC_clean/plot_spec.json"  $PY "$GLM/plot_glm_results.py"
           env PLOT_SPEC="$REPO/local_agents/OC_clean/plot_spec.json"  $PY "$GLM/plot_call_structure.py"; }
    svc(){ echo "[plots] service"; $PY "$SVC/plot_service_iso.py"; }
    case "$which" in
      agents-swe) swe ;; agents-oc) oc ;; service) svc ;;
      all) swe; oc; svc ;; *) echo "unknown plot set: $which"; exit 1 ;;
    esac ;;

  validate)
    which="${STAGE:-all}"
    va(){ echo "[validate] $1"; $PY "$GLM/validate_glm_agents.py" "$2" glm; }
    case "$which" in
      agents-swe) va SWE_clean "$REPO/local_agents/SWE_clean/data" ;;
      agents-oc)  va OC_clean  "$REPO/local_agents/OC_clean/data" ;;
      service)    $PY "$SVC/validate_service.py" ;;
      all) va SWE_clean "$REPO/local_agents/SWE_clean/data"; va OC_clean "$REPO/local_agents/OC_clean/data"; $PY "$SVC/validate_service.py" ;;
      *) echo "unknown validate set: $which"; exit 1 ;;
    esac ;;

  -h|--help|help) usage 0 ;;
  *) echo "unknown campaign: $CAMP"; echo; usage 1 ;;
esac
