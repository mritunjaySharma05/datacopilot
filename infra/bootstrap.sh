#!/bin/bash
# Runs ON THE EC2 INSTANCE at every start (via datacopilot.service -> /opt/datacopilot/run.sh).
# Uploaded to s3://<bucket>/config/ by infra/deploy.sh. Exits non-zero (and run.sh retries) until
# config, data, secrets and images are all in place.
#
# Secrets (Groq key, basic-auth hash) are written only to /run/datacopilot (tmpfs, mode 700,
# cleared at shutdown) and never printed: `set -x` is not used in this script.
set -euo pipefail
. /etc/datacopilot.env                     # BUCKET, AWS_REGION (written at first boot)
APP=/opt/datacopilot
RUN=/run/datacopilot
export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION"

mkdir -p "$APP/data"
install -d -m 700 "$RUN"

# 1. Config (not secret): compose file, Caddyfile template, site.env
for f in compose.aws.yml Caddyfile site.env; do
  aws s3 cp --only-show-errors "s3://$BUCKET/config/$f" "$APP/$f"
done
. "$APP/site.env"                          # REGISTRY, IMAGE_TAG, GROQ_MODEL, SITE_HOSTNAME, AUTH_USER

# 2. Data (read-only mount into the api container)
aws s3 sync --only-show-errors --delete "s3://$BUCKET/data/" "$APP/data/"
for f in olist.sqlite ipl_matches.csv f1_races.csv crop_production.csv; do
  [ -s "$APP/data/$f" ] || { echo "data file missing: $f"; exit 1; }
done

# 3. Secrets -> tmpfs only
param() { aws ssm get-parameter --name "$1" --with-decryption --query Parameter.Value --output text; }
umask 077
GROQ_API_KEY="$(param /datacopilot/groq_api_key)"
AUTH_HASH="$(param /datacopilot/basic_auth_hash)"
printf 'GROQ_API_KEY=%s\n' "$GROQ_API_KEY" > "$RUN/api.env"
unset GROQ_API_KEY
SITE_HOSTNAME="$SITE_HOSTNAME" AUTH_USER="$AUTH_USER" AUTH_HASH="$AUTH_HASH" python3 - "$APP/Caddyfile" "$RUN/Caddyfile" <<'PY'
import os, sys
text = open(sys.argv[1]).read()
for key in ("SITE_HOSTNAME", "AUTH_USER", "AUTH_HASH"):
    value = os.environ[key]
    if not value or any(c.isspace() for c in value):
        sys.exit(f"bad value for {key}")
    text = text.replace(f"__{key}__", value)
open(sys.argv[2], "w").write(text)
PY
unset AUTH_HASH
chmod 644 "$RUN/Caddyfile"                 # caddy reads it inside its container; holds a bcrypt hash only

# 4. Images
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
cd "$APP"
docker compose -p datacopilot --env-file site.env -f compose.aws.yml pull --quiet
docker compose -p datacopilot --env-file site.env -f compose.aws.yml up -d --remove-orphans
docker image prune -f >/dev/null
echo "DataCopilot started: image tag $IMAGE_TAG, https://$SITE_HOSTNAME"
