#!/usr/bin/env bash
set -euo pipefail

# Persist runtime env vars so login shells (e.g. via SSH) can read them.
ENV_FILE="/etc/profile.d/forwarded-runtime-env.sh"
mkdir -p "$(dirname "$ENV_FILE")"

{
  echo '# Generated at container startup to surface docker -e env vars'
  for var in HF_UPLOAD_REPO_ID WANDB_API_KEY HF_TOKEN WANDB_RUN; do
    value="${!var:-}"
    if [[ -n "$value" ]]; then
      printf 'export %s=%q\n' "$var" "$value"
    else
      printf 'unset %s\n' "$var"
    fi
  done
} > "$ENV_FILE"
chmod 0644 "$ENV_FILE"

exec "$@"
