#!/usr/bin/env bash
# oc_cgroup_watcher.sh <docker-full-id> — harness/tool separation inside the OC container.
# The wildclawbench image runs agent+gateway+tools in ONE container. This watcher creates two
# child cgroups under the container scope and sorts processes by comm:
#   node / openclaw* / bun -> agent    (agent+gateway; the gateway's comm is
#                                       'openclaw-gatewa', NOT node — verified 2026-07-08)
#   everything else        -> toolexec (chromium, python, converters, shells, ...)
# Children inherit their parent's cgroup at fork, so tool processes are BORN in /agent (their
# spawner is node) — the loop must therefore keep sweeping /agent, not just the scope root.
# Leakage = first <POLL_S of each tool process; quantified post-hoc from record samples.
# MUST run as root (cgroup files under system.slice). Pin to housekeeping CPUs from the caller.
set -o pipefail
SCOPE="$1"          # absolute cgroup dir of the container, e.g. /sys/fs/cgroup/measured.slice/docker-<id>.scope
POLL_S="${POLL_S:-0.02}"   # freshly-cloned tool procs carry comm 'node' until exec — short
                           # residency is the whole game (E4 measured 19.8% at 0.05 on the
                           # near-idle calendar episode, dominated by one pip spawn burst)
[ -d "$SCOPE" ] || { echo "watcher: no scope $SCOPE" >&2; exit 1; }
mkdir -p "$SCOPE/agent" "$SCOPE/toolexec" || exit 1

move(){ echo "$1" > "$2/cgroup.procs" 2>/dev/null; }   # fails harmlessly if pid died

while [ -d "$SCOPE" ]; do
  # scope root: freshly attached by docker (exec) — sort everything out of it
  for pid in $(cat "$SCOPE/cgroup.procs" 2>/dev/null); do
    c=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$c" in node|openclaw*|bun) move "$pid" "$SCOPE/agent" ;; *) move "$pid" "$SCOPE/toolexec" ;; esac
  done
  # /agent: non-agent children forked by the agent are born here — sweep them to toolexec
  for pid in $(cat "$SCOPE/agent/cgroup.procs" 2>/dev/null); do
    c=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$c" in node|openclaw*|bun) : ;; *) move "$pid" "$SCOPE/toolexec" ;; esac
  done
  # /toolexec: the SYMMETRIC sweep — container-startup procs get classified pre-exec
  # (comm sh/npm) and land here, then exec into node/openclaw; without this sweep they
  # would be stuck on the wrong side forever (one-way bug, verified 2026-07-08)
  for pid in $(cat "$SCOPE/toolexec/cgroup.procs" 2>/dev/null); do
    c=$(cat "/proc/$pid/comm" 2>/dev/null) || continue
    case "$c" in node|openclaw*|bun) move "$pid" "$SCOPE/agent" ;; esac
  done
  sleep "$POLL_S"
done
