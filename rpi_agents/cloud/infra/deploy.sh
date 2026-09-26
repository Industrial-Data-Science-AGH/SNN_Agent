#!/usr/bin/env bash
# Deploy the SNN backend to Azure with the az CLI. Idempotent: run it again to roll out a new image or a changed template.
#
#   az login && az account set --subscription <id>
#   PYTHON=/path/to/venv/bin/python ./rpi_agents/cloud/infra/deploy.sh
#
# Needs: az (with the containerapp extension not required), openssl, and a Python with requirements-backend.lock
# installed (only to hash the operator password). Run it from the repository root.
#
# Environment (all optional):
#   RG, LOCATION            resource group and region (default rg-snn-backend-dev, swedencentral)
#   PREFIX                  name prefix (default snnbe)
#   IMAGE_TAG               image tag to build and deploy (default: short git sha)
#   MAIL_ENV_FILE           a KEY=VALUE file with GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO (the old agent's format).
#                           Without it, mail is not configured and the secrets are not touched.
#   POLICY                  manual-review-only-v1 (default) or armed-glass-and-person-v1
#   ENVIRONMENT_ID          resource id of an existing Container Apps environment to join (needed when the subscription
#                           already has one and allows no more; the region must match it). Default: create one.
#   EXTRA_PARAMS            more `key=value` template parameters, space separated
#
# Secrets never appear in this script's output, in a command line or in the template: they go from a file or stdin into
# Key Vault. A generated operator password is stored in the vault as `operator-password` (read it with
# `az keyvault secret show --vault-name <vault> --name operator-password --query value -o tsv`).
set -euo pipefail

RG=${RG:-rg-snn-backend-dev}
LOCATION=${LOCATION:-swedencentral}
PREFIX=${PREFIX:-snnbe}
PYTHON=${PYTHON:-python3}
POLICY=${POLICY:-manual-review-only-v1}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}
TEMPLATE=rpi_agents/cloud/infra/main.bicep
TAGS="project=snn-agent component=backend environment=dev"

log() { printf '[deploy] %s\n' "$*" >&2; }
out() { az deployment group show -g "$RG" -n "$1" --query "properties.outputs.$2.value" -o tsv; }

put_secret() { # put_secret <name>   (value on stdin; never echoed, never on a command line)
  local file
  file=$(mktemp) && chmod 600 "$file"
  cat >"$file"
  for attempt in 1 2 3 4 5 6 7 8; do # a fresh role assignment takes a while to reach the vault
    if az keyvault secret set --vault-name "$VAULT" --name "$1" --file "$file" --encoding utf-8 -o none 2>/dev/null; then
      rm -f "$file"; log "secret $1 stored"; return 0
    fi
    sleep 15
  done
  rm -f "$file"; log "could not store secret $1 (does the signed-in user have Key Vault Secrets Officer?)"; return 1
}

secret_exists() { az keyvault secret show --vault-name "$VAULT" --name "$1" --query id -o tsv >/dev/null 2>&1; }

log "resource group $RG in $LOCATION"
az group create -n "$RG" -l "$LOCATION" --tags $TAGS -o none
ADMIN=$(az ad signed-in-user show --query id -o tsv)

log "phase 1: infrastructure without the apps"
# shellcheck disable=SC2086
az deployment group create -g "$RG" -n snn-base -f "$TEMPLATE" -o none \
  -p prefix="$PREFIX" deployApps=false adminPrincipalId="$ADMIN" policy="$POLICY" existingEnvironmentId="${ENVIRONMENT_ID:-}" ${EXTRA_PARAMS:-}
VAULT=$(out snn-base vaultName); REGISTRY=$(out snn-base registryName)

log "building the image in the registry ($REGISTRY, tag $IMAGE_TAG)"
az acr build -r "$REGISTRY" -t "snn-backend:$IMAGE_TAG" -f rpi_agents/cloud/infra/Dockerfile --no-logs . -o none \
  || { log "az acr build failed; is ACR Tasks blocked on this subscription? Falling back to a local docker build"; \
       az acr login -n "$REGISTRY" >/dev/null && docker build -f rpi_agents/cloud/infra/Dockerfile -t "$REGISTRY.azurecr.io/snn-backend:$IMAGE_TAG" . \
       && docker push "$REGISTRY.azurecr.io/snn-backend:$IMAGE_TAG"; }

if ! secret_exists operator-password-hash; then
  log "creating the operator password"
  PASSWORD=$(openssl rand -base64 24 | tr -d '\n')
  printf '%s' "$PASSWORD" | put_secret operator-password
  printf '%s\n' "$PASSWORD" | PYTHONPATH=. "$PYTHON" -m rpi_agents.cloud.app.admin hash-password | tr -d '\n' | put_secret operator-password-hash
  unset PASSWORD
fi

MAIL=true
if [ -n "${MAIL_ENV_FILE:-}" ]; then
  get() { grep -E "^$1=" "$MAIL_ENV_FILE" | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' | tr -d '\r\n'; }
  get GMAIL_USER | put_secret smtp-user
  get GMAIL_USER | put_secret smtp-from
  get GMAIL_APP_PASSWORD | tr -d " " | put_secret smtp-password
  get ALERT_TO | put_secret alert-recipients
elif ! secret_exists smtp-password; then
  log "no MAIL_ENV_FILE and no mail secrets in the vault: deploying without e-mail"; MAIL=false
fi

log "phase 2: the container apps"
# shellcheck disable=SC2086
az deployment group create -g "$RG" -n snn-apps -f "$TEMPLATE" -o none \
  -p prefix="$PREFIX" deployApps=true adminPrincipalId="$ADMIN" policy="$POLICY" mailEnabled="$MAIL" existingEnvironmentId="${ENVIRONMENT_ID:-}" \
     imageName="snn-backend:$IMAGE_TAG" ${EXTRA_PARAMS:-}
HOST=$(out snn-apps apiHost)

log "waiting for https://$HOST/healthz"
for _ in $(seq 1 40); do
  if curl -fsS -m 5 "https://$HOST/healthz" >/dev/null 2>&1; then log "healthy"; break; fi
  sleep 6
done
printf 'api_host=%s\nvault=%s\nregistry=%s\nstorage=%s\nvision=%s (%s)\nimage=snn-backend:%s\n' \
  "$HOST" "$VAULT" "$REGISTRY" "$(out snn-apps storageAccount)" "$(out snn-apps visionDeployment)" "$(out snn-apps visionEndpoint)" "$IMAGE_TAG"
