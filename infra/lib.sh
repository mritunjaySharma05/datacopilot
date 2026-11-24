#!/usr/bin/env bash
# Shared settings and helpers for the infra/*.sh scripts (source it; it calls no AWS APIs itself).
# Works in bash on macOS, Linux, and Git Bash / WSL on Windows.
set -euo pipefail

REGION=ap-southeast-2            # the project account is locked to Sydney
STACK=datacopilot
CADDY_IMAGE=caddy:2.11.9
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""

info() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need() {  # need CMD... : fail early if a tool is missing
  local cmd
  for cmd in "$@"; do command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' not found on PATH"; done
}

env_get() {  # env_get KEY [DEFAULT] : value from .env (last assignment wins); never printed by the scripts
  local key=$1 default=${2:-} line value
  if [ -f "$ENV_FILE" ]; then
    line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$ENV_FILE" | tail -n 1 || true)
  else
    line=""
  fi
  if [ -z "$line" ]; then printf '%s' "$default"; return 0; fi
  value=${line#*=}
  value=${value%$'\r'}                                  # Windows line endings
  value=${value#"${value%%[![:space:]]*}"}               # leading spaces
  case $value in
    \"*) value=${value#\"}; value=${value%%\"*} ;;      # "quoted" (rest of line ignored)
    \'*) value=${value#\'}; value=${value%%\'*} ;;      # 'quoted'
    *)   value=${value%%[[:space:]]#*}                   # unquoted: drop " # comment"
         value=${value%"${value##*[![:space:]]}"} ;;    # trailing spaces
  esac
  if [ -z "$value" ]; then value=$default; fi
  printf '%s' "$value"
}

require_env() {  # require_env KEY... : fail with a clear message if a .env setting is empty
  local key
  for key in "$@"; do [ -n "$(env_get "$key")" ] || die "set $key in $ENV_FILE (see docs/notes/handoff.md)"; done
}

stack_output() {  # stack_output NAME : one output value of the stack (empty if missing)
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null | tr -d '\r'
}

stack_exists() {
  aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1
}
