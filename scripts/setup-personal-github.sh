#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GH_CONFIG_DIR="${REPO_ROOT}/.gh"
mkdir -p "${GH_CONFIG_DIR}"

cd "${REPO_ROOT}"

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "Sign in with your personal GitHub account (RohanRJ389)."
  gh auth login -h github.com -p https -s repo
fi

if ! gh repo view RohanRJ389/hackerrank-orchestrate-september26 >/dev/null 2>&1; then
  gh repo create RohanRJ389/hackerrank-orchestrate-september26 \
    --private \
    --source=. \
    --remote=origin \
    --push
else
  git remote set-url origin https://github.com/RohanRJ389/hackerrank-orchestrate-september26.git
  git push -u origin main
fi

echo "Done. origin -> https://github.com/RohanRJ389/hackerrank-orchestrate-september26.git"
