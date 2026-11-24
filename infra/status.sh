#!/usr/bin/env bash
# Show the deployment state: stack, instance, URL check, alarm, remaining AWS credits.
#   infra/status.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need aws curl
if ! stack_exists; then info "stack $STACK: not deployed"; exit 0; fi
ID=$(stack_output InstanceId)
URL=$(stack_output Url)
printf '%-12s %s\n' "stack" "$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text | tr -d '\r')"
printf '%-12s %s %s\n' "instance" "$ID" "$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text | tr -d '\r')"
printf '%-12s %s\n' "public ip" "$(stack_output PublicIp)"
printf '%-12s %s (HTTP %s; 401 = up, login required)\n' "url" "$URL" "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/" || true)"
printf '%-12s %s\n' "ssm agent" "$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$ID" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null | tr -d '\r')"
printf '%-12s %s\n' "alarm" "$(aws cloudwatch describe-alarms --alarm-names datacopilot-api-5xx --query 'MetricAlarms[0].StateValue' --output text | tr -d '\r')"
printf '%-12s %s USD\n' "credits" "$(aws freetier get-account-plan-state --region us-east-1 --query 'accountPlanRemainingCredits.amount' --output text 2>/dev/null | tr -d '\r' || echo '?')"
