#!/usr/bin/env bash
# Delete EVERYTHING the deployment created (asks you to type the stack name first).
#   infra/delete.sh
# Removes: S3 data/config, the stack (instance, disk, Elastic IP, ECR repositories and images,
# log groups, alarm, SNS topic, budget, IAM role, security group) and the two SSM parameters.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need aws
printf 'This deletes the whole DataCopilot deployment in %s. Type "%s" to continue: ' "$REGION" "$STACK"
read -r answer
[ "$answer" = "$STACK" ] || die "cancelled"

if stack_exists; then
  BUCKET=$(stack_output BucketName)
  if [ -n "$BUCKET" ] && aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    info "emptying s3://$BUCKET"
    aws s3 rm "s3://$BUCKET" --recursive --only-show-errors
  fi
  info "deleting stack $STACK (5-10 minutes)"
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
else
  info "stack $STACK not found"
fi
info "deleting SSM parameters"
aws ssm delete-parameters --names /datacopilot/groq_api_key /datacopilot/basic_auth_hash >/dev/null
info "done. Nothing from the deployment is left. (The DuckDNS name still points at the released IP.)"
