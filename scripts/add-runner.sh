#!/usr/bin/env bash
# Register a GitHub Actions self-hosted runner for one repo on the Zephyrus.
# Usage: scripts/add-runner.sh <repo-name> [label]     (run from a PC with gh logged in and ssh zephyrus working)
# Example: scripts/add-runner.sh ai-hub ai-hub
set -euo pipefail
repo=${1:?repo name, e.g. ai-hub}
label=${2:-$repo}
owner=Zaher503
token=$(gh api -X POST "repos/$owner/$repo/actions/runners/registration-token" --jq .token)
ssh zephyrus bash -s -- "$owner" "$repo" "$label" "$token" <<'REMOTE'
set -euo pipefail
owner=$1 repo=$2 label=$3 token=$4
dir=/opt/actions-runner-$repo
if [ ! -d "$dir" ]; then
  sudo install -d -o zaher -g zaher "$dir"
  # Reuse the downloaded runner binaries, without the first repo's registration
  tar -C /opt/actions-runner --exclude='.runner*' --exclude=.credentials --exclude=.credentials_rsaparams \
      --exclude=_work --exclude=_diag --exclude=.service -cf - . | tar -C "$dir" -xf -
fi
cd "$dir"
./config.sh --unattended --replace --url "https://github.com/$owner/$repo" --token "$token" --name zephyrus --labels "$label"
sudo ./svc.sh install zaher
sudo ./svc.sh start
sudo ./svc.sh status | head -5
REMOTE
