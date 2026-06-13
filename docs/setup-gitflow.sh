#!/usr/bin/env bash
# One-time GitHub branch-protection setup for clawde's Gitflow (see ADR-0002).
# Requires the GitHub CLI (`gh auth login`) and a pushed remote with main+develop.
#
# Usage: bash docs/setup-gitflow.sh <owner/repo>
set -euo pipefail

REPO="${1:?usage: bash docs/setup-gitflow.sh <owner/repo>}"

echo "Protecting main on $REPO (PR required, CI + guard must pass, no direct push)…"
gh api -X PUT "repos/$REPO/branches/main/protection" \
  -H "Accept: application/vnd.github+json" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=lint · type · test (ubuntu-latest)" \
  -f "required_status_checks[contexts][]=Branch naming" \
  -F "enforce_admins=true" \
  -F "required_pull_request_reviews[required_approving_review_count]=0" \
  -F "restrictions=null" >/dev/null

echo "Protecting develop on $REPO (CI + guard must pass on PRs)…"
gh api -X PUT "repos/$REPO/branches/develop/protection" \
  -H "Accept: application/vnd.github+json" \
  -f "required_status_checks[strict]=true" \
  -f "required_status_checks[contexts][]=lint · type · test (ubuntu-latest)" \
  -f "required_status_checks[contexts][]=Branch naming" \
  -F "enforce_admins=false" \
  -F "required_pull_request_reviews=null" \
  -F "restrictions=null" >/dev/null

echo "Done. Verify in the repo's Settings → Branches."
