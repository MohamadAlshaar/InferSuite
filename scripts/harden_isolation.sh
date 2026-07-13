#!/usr/bin/env bash
# Optional BOOT-TIME hardening of the measurement isolation.
#
# The runtime isolation (measured.slice AllowedCPUs, governor/no_turbo/THP/NMI)
# is "soft": the measured cores still take their local timer tick and host
# unpinnable per-CPU kernel threads. This script adds GRUB kernel parameters
# that make the MEASURED cores scheduler-isolated, tickless, and RCU-offloaded:
#
#     isolcpus=<measured>   nohz_full=<measured>   rcu_nocbs=<measured>
#
# It COMPLEMENTS the runtime cpuset isolation (you still pin the workload with
# measured.slice); it does not replace it. The house cores (0-1,12-13) remain
# the housekeeping/tick CPUs, which nohz_full requires.
#
# REQUIRES A REBOOT to take effect. Fully reversible with --off. It only affects
# FUTURE captures; already-banked data is unchanged.
#
# Usage:  sudo scripts/harden_isolation.sh [--on|--off|--status]
#         CPUS_MEASURED=2-11,14-23 sudo scripts/harden_isolation.sh --on   # override range
set -euo pipefail

CPUS_MEASURED="${CPUS_MEASURED:-2-11,14-23}"   # must match campaign.conf
GRUB="/etc/default/grub"
KEY="GRUB_CMDLINE_LINUX_DEFAULT"
PARAMS=(isolcpus nohz_full rcu_nocbs)
MODE="${1:-}"

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
[ -f "$GRUB" ] || { echo "no $GRUB — this looks like a non-GRUB system; aborting."; exit 1; }

# current value of the cmdline key (last definition wins)
cur="$(grep -E "^${KEY}=" "$GRUB" | tail -1 | sed -E "s/^${KEY}=\"?(.*[^\"])\"?\$/\1/" || true)"
# strip any existing copies of our params so we never duplicate/conflict
clean="$cur"
for p in "${PARAMS[@]}"; do clean="$(sed -E "s/(^| )${p}=[^ ]*//g" <<<"$clean")"; done
clean="$(tr -s ' ' <<<"$clean" | sed -E 's/^ | $//g')"

case "$MODE" in
  --status)
    echo "${KEY}=\"$cur\""
    for p in "${PARAMS[@]}"; do grep -qE "(^| )${p}=" <<<"$cur" && echo "  hardening: ON ($p present)" && break; done \
      || echo "  hardening: OFF"
    exit 0 ;;
  --on)  newval="$(tr -s ' ' <<<"$clean isolcpus=$CPUS_MEASURED nohz_full=$CPUS_MEASURED rcu_nocbs=$CPUS_MEASURED" | sed -E 's/^ | $//g')" ;;
  --off) newval="$clean" ;;
  *) echo "usage: sudo $0 [--on|--off|--status]"; exit 1 ;;
esac

cp -n "$GRUB" "${GRUB}.pre-isoharden.bak"   # one-time backup
echo "backup: ${GRUB}.pre-isoharden.bak"
if grep -qE "^${KEY}=" "$GRUB"; then
  sed -i -E "s|^${KEY}=.*|${KEY}=\"${newval}\"|" "$GRUB"
else
  echo "${KEY}=\"${newval}\"" >> "$GRUB"
fi
echo "set    ${KEY}=\"${newval}\""
update-grub
echo
echo ">>> REBOOT REQUIRED for this to take effect."
echo ">>> after reboot, verify with:  cat /proc/cmdline   (should show the params)"
echo ">>> revert any time:            sudo $0 --off   &&  reboot"
