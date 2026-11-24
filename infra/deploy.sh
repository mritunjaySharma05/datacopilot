#!/usr/bin/env bash
# Create or update the whole AWS deployment (idempotent; run it again to ship a new version).
#   infra/deploy.sh
# Reads settings and secrets from .env (never printed; see docs/notes/handoff.md):
#   GROQ_API_KEY, BASIC_AUTH_PASSWORD     secrets -> SSM Parameter Store (SecureString)
#   DUCKDNS_TOKEN                          used locally to point the hostname at the Elastic IP
#   SITE_HOSTNAME, ALARM_EMAIL             e.g. myname.duckdns.org, you@example.com
#   BASIC_AUTH_USER (demo), GROQ_MODEL (openai/gpt-oss-120b), BUDGET_USD (30)   optional
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$ROOT"
need aws docker git curl

# ---------------------------------------------------------------- settings
require_env GROQ_API_KEY BASIC_AUTH_PASSWORD SITE_HOSTNAME ALARM_EMAIL
SITE_HOSTNAME=$(env_get SITE_HOSTNAME)
ALARM_EMAIL=$(env_get ALARM_EMAIL)
AUTH_USER=$(env_get BASIC_AUTH_USER demo)
GROQ_MODEL=$(env_get GROQ_MODEL openai/gpt-oss-120b)
BUDGET_USD=$(env_get BUDGET_USD 30)
[ "${#AUTH_USER}" -ge 1 ] && [[ $AUTH_USER =~ ^[A-Za-z0-9._-]+$ ]] || die "BASIC_AUTH_USER: letters, digits, . _ - only"
[ "$(env_get BASIC_AUTH_PASSWORD | wc -c | tr -d " ")" -ge 12 ] || die "BASIC_AUTH_PASSWORD must be at least 12 characters"
[[ $SITE_HOSTNAME =~ ^[a-z0-9-]+(\.[a-z0-9-]+)+$ ]] || die "SITE_HOSTNAME looks wrong: $SITE_HOSTNAME"

for f in olist.sqlite ipl_matches.csv ipl_deliveries.csv f1_races.csv crop_production.csv; do
  [ -s "data/processed/$f" ] || die "data/processed/$f missing: run scripts/prepare_data.py first"
done

TAG=$(git rev-parse --short=12 HEAD)
if ! git diff --quiet HEAD -- src models docker; then TAG="$TAG-dirty-$(date +%Y%m%d%H%M%S)"; fi
ACCOUNT=$(aws sts get-caller-identity --query Account --output text | tr -d '\r')
info "account $ACCOUNT, region $REGION, stack $STACK, image tag $TAG"

TMP="$ROOT/infra/.secrets-tmp"       # gitignored; removed on exit
rm -rf "$TMP"; (umask 077; mkdir -p "$TMP")
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- 1. stack
info "1/6 CloudFormation stack"
aws cloudformation validate-template --template-body file://infra/datacopilot.yaml >/dev/null
STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null | tr -d '\r' || true)
if [ "$STATUS" = "ROLLBACK_COMPLETE" ]; then   # a failed first create cannot be updated; it holds no resources
  info "    removing the failed first attempt (ROLLBACK_COMPLETE)"
  aws cloudformation delete-stack --stack-name "$STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK"
fi
aws cloudformation deploy --stack-name "$STACK" --template-file infra/datacopilot.yaml \
  --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "SiteHostname=$SITE_HOSTNAME" "AlarmEmail=$ALARM_EMAIL" "BudgetAmountUsd=$BUDGET_USD"
BUCKET=$(stack_output BucketName)
REGISTRY=$(stack_output Registry)
INSTANCE_ID=$(stack_output InstanceId)
PUBLIC_IP=$(stack_output PublicIp)
[ -n "$BUCKET" ] && [ -n "$INSTANCE_ID" ] || die "stack outputs missing"

# ---------------------------------------------------------------- 2. secrets
info "2/6 secrets -> SSM Parameter Store (SecureString)"
json_escape() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '%s' "$s"; }
put_secret() {  # put_secret NAME VALUE : via a 0600 temp file, never on the command line
  (umask 077; printf '{"Name":"%s","Value":"%s","Type":"SecureString","Overwrite":true}' \
    "$1" "$(json_escape "$2")" > "$TMP/param.json")
  aws ssm put-parameter --cli-input-json "file://infra/.secrets-tmp/param.json" >/dev/null
  rm -f "$TMP/param.json"
}
put_secret /datacopilot/groq_api_key "$(env_get GROQ_API_KEY)"
AUTH_HASH=$(env_get BASIC_AUTH_PASSWORD | docker run --rm -i "$CADDY_IMAGE" caddy hash-password | tr -d '\r')
[[ $AUTH_HASH == \$2* ]] || die "could not hash the basic-auth password"
put_secret /datacopilot/basic_auth_hash "$AUTH_HASH"

# ---------------------------------------------------------------- 3. images
info "3/6 build and push linux/amd64 images"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
for target in api ui; do
  docker buildx build --platform linux/amd64 --provenance=false -f docker/Dockerfile --target "$target" \
    -t "$REGISTRY/datacopilot/$target:$TAG" --push --quiet . >/dev/null
  info "    pushed datacopilot/$target:$TAG"
done

# ---------------------------------------------------------------- 4. data and config
info "4/6 data and config -> s3://$BUCKET"
aws s3 sync data/processed "s3://$BUCKET/data/" --delete --only-show-errors --exclude ".gitkeep"
printf 'REGISTRY=%s\nIMAGE_TAG=%s\nGROQ_MODEL=%s\nAWS_REGION=%s\nSITE_HOSTNAME=%s\nAUTH_USER=%s\n' \
  "$REGISTRY" "$TAG" "$GROQ_MODEL" "$REGION" "$SITE_HOSTNAME" "$AUTH_USER" > "$TMP/site.env"
aws s3 cp --only-show-errors "$TMP/site.env" "s3://$BUCKET/config/site.env"
for f in compose.aws.yml Caddyfile bootstrap.sh; do
  aws s3 cp --only-show-errors "infra/$f" "s3://$BUCKET/config/$f"
done

# ---------------------------------------------------------------- 5. DNS
info "5/6 hostname $SITE_HOSTNAME -> $PUBLIC_IP"
DUCKDNS_TOKEN=$(env_get DUCKDNS_TOKEN)
if [ -n "$DUCKDNS_TOKEN" ] && [[ $SITE_HOSTNAME == *.duckdns.org ]]; then
  # token passed to curl on stdin (-K -), not on the command line
  reply=$(printf 'url = "https://www.duckdns.org/update?domains=%s&token=%s&ip=%s"\n' \
    "${SITE_HOSTNAME%.duckdns.org}" "$DUCKDNS_TOKEN" "$PUBLIC_IP" | curl -fsS --max-time 20 -K -)
  [ "$reply" = "OK" ] || die "DuckDNS update failed (reply: $reply); check DUCKDNS_TOKEN and SITE_HOSTNAME"
else
  info "    no DUCKDNS_TOKEN: point $SITE_HOSTNAME at $PUBLIC_IP yourself"
fi

# ---------------------------------------------------------------- 6. (re)start the app
info "6/6 start the app on $INSTANCE_ID"
PING=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null | tr -d '\r' || true)
if [ "$PING" = "Online" ]; then
  aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
    --comment "datacopilot deploy $TAG" --parameters 'commands=["systemctl restart datacopilot"]' >/dev/null
  info "    restart requested"
else
  info "    first boot: the instance starts the app by itself once data and images are there"
fi

URL="https://$SITE_HOSTNAME"
info "waiting for $URL (certificate and containers can take a few minutes)"
for _ in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/" || true)
  if [ "$code" = "401" ]; then
    info "live: $URL (log in as '$AUTH_USER'). Confirm the SNS email to get alarm notifications."
    exit 0
  fi
  sleep 15
done
die "$URL did not answer with the login prompt yet; check infra/status.sh and the /datacopilot/* log groups"
