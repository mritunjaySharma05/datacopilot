#!/usr/bin/env bash
# Stop the instance (everything is kept; only the disk and the Elastic IP bill, about $0.17/day).
#   infra/stop.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need aws
ID=$(stack_output InstanceId)
[ -n "$ID" ] || die "stack $STACK not found"
aws ec2 stop-instances --instance-ids "$ID" >/dev/null
info "stopping $ID ..."
aws ec2 wait instance-stopped --instance-ids "$ID"
info "stopped. Start again with infra/start.sh"
