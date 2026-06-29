#!/usr/bin/env bash
# Terminate the two instances. Add --all to also delete the free scaffolding (SG/key).
set -uo pipefail
cd "$(dirname "$0")"; . ./env.sh
[ -f state.env ] && . ./state.env || true

echo "=== terminating instances tagged proj=agentic-uw2 in $R ==="
IDS=$(aws ec2 describe-instances --region "$R" \
  --filters "Name=tag:proj,Values=agentic-uw2" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$IDS" ] && [ "$IDS" != "None" ]; then
  aws ec2 terminate-instances --region "$R" --instance-ids $IDS --query 'TerminatingInstances[].{id:InstanceId,state:CurrentState.Name}' --output text
  aws ec2 wait instance-terminated --region "$R" --instance-ids $IDS && echo "terminated."
else echo "no instances running."; fi
rm -f state.env

if [ "${1:-}" = "--all" ]; then
  echo "=== removing free scaffolding (SG + key) ==="
  aws ec2 delete-security-group --region "$R" --group-id "$SG" 2>/dev/null && echo "SG deleted" || echo "SG delete skipped (in use?)"
  aws ec2 delete-key-pair --region "$R" --key-name "$KEY" 2>/dev/null && echo "key deleted"
fi
echo "=== verify nothing left ==="
aws ec2 describe-instances --region "$R" --filters "Name=tag:proj,Values=agentic-uw2" "Name=instance-state-name,Values=running,pending" --query 'Reservations[].Instances[].InstanceId' --output text
echo "(empty above = clean)"
