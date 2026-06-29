# Shared perf helpers for the agentic suite. Source after perf_events.sh.
# Baked-in fixes: validate perf binary; sudo only when needed; HARD-FAIL on empty/zero
# output (no silent zeros); SIGINT+wait flush (not fixed sleep); effective freq (not 4.6GHz).

# Resolve a WORKING perf binary. The /usr/bin/perf WRAPPER refuses to run when its
# linux-tools build doesn't match the running kernel (common on OEM/HWE kernels:
# "WARNING: perf not found for kernel X"), and a kernel-specific build under
# /usr/lib/linux-tools-*/ may be present but for a different version. So we don't trust
# existence — we TEST each candidate by making it actually count 'cycles' on a tiny
# workload and return the first that produces a real number. Cached in _PERF_BIN.
perf_bin() {
  [ -n "${_PERF_BIN:-}" ] && { echo "$_PERF_BIN"; return; }
  local c out
  for c in "${PERF_HOST_BIN:-}" /usr/lib/linux-tools-*/perf /usr/bin/perf perf; do
    [ -n "$c" ] || continue
    command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
    out="$("$c" stat -e cycles -- awk 'BEGIN{for(i=0;i<5000000;i++)s+=i}' 2>&1)" || continue
    if printf '%s' "$out" | grep -qiE '[0-9][0-9,.]*[[:space:]]+cycles'; then
      _PERF_BIN="$c"; echo "$c"; return
    fi
  done
  echo "FATAL: no WORKING perf binary found (set PERF_HOST_BIN to a build matching $(uname -r))" >&2
  return 1
}

# "sudo -n" if we're not root AND the target isn't ours; else "".
# Pass a target uid (optional) to decide; default: sudo if not root.
perf_sudo() {
  [ "$(id -u)" = "0" ] && { echo ""; return; }
  echo "sudo -n"
}

# Ensure perf can read HW counters (idempotent; needs sudo on a tightened box).
perf_enable() {
  local s; s="$(perf_sudo)"
  $s sysctl -q kernel.perf_event_paranoid=-1 2>/dev/null || true
  $s bash -c 'echo 4096 > /proc/sys/kernel/perf_event_mlock_kb' 2>/dev/null || true
}

# Effective average core frequency in Hz (mean of scaling_cur_freq, kHz->Hz).
# Falls back to cpuinfo_max_freq, then 0 (caller must handle 0).
eff_freq_hz() {
  local sum=0 n=0 f
  for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    [ -r "$f" ] || continue; sum=$((sum + $(cat "$f"))); n=$((n+1))
  done
  if [ "$n" -gt 0 ]; then echo $(( sum / n * 1000 )); return; fi
  local m=/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
  [ -r "$m" ] && echo $(( $(cat "$m") * 1000 )) || echo 0
}

# HARD-FAIL check: a perf -o file must exist and contain a nonzero 'cycles' (or task-clock).
# Usage: assert_perf_ok <file> <label>  -> exits 1 if the pass silently produced nothing.
assert_perf_ok() {
  local f="$1" lbl="${2:-pass}"
  [ -s "$f" ] || { echo "FATAL[$lbl]: perf output missing/empty: $f" >&2; return 1; }
  # accept either CSV (-x,) or human format; look for a cycles line with a >0 number
  if ! grep -qiE '(^|,)[[:space:]]*[1-9][0-9,]*[[:space:]]*,?[[:space:]]*cycles|[1-9][0-9,]*[[:space:]]+cycles' "$f"; then
    echo "FATAL[$lbl]: no nonzero 'cycles' in $f (perf failed? wrong scope? counter unsupported?)" >&2
    echo "---- $f ----" >&2; sed -n '1,15p' "$f" >&2; return 1
  fi
  return 0
}

# Run ONE aggregate perf pass for a fixed DURATION over a scope, stdin CLOSED (no prompt hang).
# Usage: perf_aggregate "<events>" "<scope args>" <seconds> <outfile> <label>
#   scope args examples:  "-a"   |   "-a -G <cgroup>"   |   "-p <pid>"
perf_aggregate() {
  local events="$1" scope="$2" secs="$3" out="$4" lbl="${5:-pass}"
  local PERF SUDO; PERF="$(perf_bin)" || return 1; SUDO="$(perf_sudo)"
  rm -f "$out"
  # shellcheck disable=SC2086
  $SUDO "$PERF" stat -e "$events" $scope -o "$out" </dev/null -- sleep "$secs" 2>>"$out".err
  assert_perf_ok "$out" "$lbl" || return 1
}

# Find the cgroup path (relative, for perf -G) of a process by PID.
cgroup_of_pid() {
  local pid="$1" s; s="$(perf_sudo)"
  $s cat "/proc/$pid/cgroup" 2>/dev/null | awk -F: '{print $3}' | grep -E 'kubepods|docker|scope' | head -1 | sed 's#^/##'
}
