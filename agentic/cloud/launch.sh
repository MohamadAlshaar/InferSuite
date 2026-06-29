#!/usr/bin/env bash
# Launch the two us-west-2 boxes (THIS is the only step that costs money).
#   GPU : g6e.2xlarge (1x L40S 48GB, Nitro)  -> vLLM 32B + ncu
#   CPU : c7i.metal-24xl (bare metal, PMU)    -> agents + perf/TMA
# Both auto-bootstrap via user-data. Writes agentic/cloud/state.env with IDs/IPs.
set -euo pipefail
cd "$(dirname "$0")"; . ./env.sh
echo "Launching in $R/$AZ — GPU=$GPU_TYPE CPU=$CPU_TYPE"

GPU_ID=$(aws ec2 run-instances --region "$R" --instance-type "$GPU_TYPE" --image-id "$AMI" \
  --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$SUBNET" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":400,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://userdata_gpu.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=agentic-gpu-l40s},{Key=proj,Value=agentic-uw2}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "GPU instance: $GPU_ID"

CPU_ID=$(aws ec2 run-instances --region "$R" --instance-type "$CPU_TYPE" --image-id "$AMI" \
  --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$SUBNET" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":400,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://userdata_metal.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=agentic-cpu-metal},{Key=proj,Value=agentic-uw2}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "CPU instance: $CPU_ID"

aws ec2 wait instance-running --region "$R" --instance-ids "$GPU_ID" "$CPU_ID"
GPU_IP=$(aws ec2 describe-instances --region "$R" --instance-ids "$GPU_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
GPU_PRIV=$(aws ec2 describe-instances --region "$R" --instance-ids "$GPU_ID" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
CPU_IP=$(aws ec2 describe-instances --region "$R" --instance-ids "$CPU_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat > state.env <<EOF
export GPU_ID=$GPU_ID
export CPU_ID=$CPU_ID
export GPU_IP=$GPU_IP
export GPU_PRIV=$GPU_PRIV   # use this from the metal box to reach vLLM (in-VPC)
export CPU_IP=$CPU_IP
EOF
echo "=== LAUNCHED (billing now). state.env written ==="
cat state.env
echo "GPU ssh: ssh -i $PEM ubuntu@$GPU_IP   (bootstrap log: /var/log/agentic_setup.log, ready flag: /opt/agentic/READY)"
echo "CPU ssh: ssh -i $PEM ubuntu@$CPU_IP"
echo "From the metal box, vLLM will be at http://$GPU_PRIV:8000/v1"
