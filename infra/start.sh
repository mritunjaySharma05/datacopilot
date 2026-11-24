#!/usr/bin/env bash
# Start the stopped instance; the datacopilot service starts the app by itself (about 2-3 min).
#   infra/start.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need aws curl
ID=$(stack_output InstanceId)
URL=$(stack_output Url)
[ -n "$ID" ] || die "stack $STACK not found"
aws ec2 start-instances --instance-ids "$ID" >/dev/null
info "starting $ID ..."
aws ec2 wait instance-running --instance-ids "$ID"
info "running; waiting for $URL"
for _ in $(seq 1 40); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/" || true)" = "401" ] && { info "live: $URL"; exit 0; }
  sleep 15
done
die "$URL not answering yet; check infra/status.sh"
