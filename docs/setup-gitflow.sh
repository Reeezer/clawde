#!/usr/bin/env bash
# One-time GitHub branch-protection setup for clawde's Gitflow (see ADR-0002).
# Requires the GitHub CLI (`gh auth login`) and the repo pushed with main+develop.
# clawde applies this at setup; rerun to reset protection to the baseline below.
#
# Usage: bash docs/setup-gitflow.sh <owner/repo>
set -euo pipefail

REPO="${1:?usage: bash docs/setup-gitflow.sh <owner/repo>}"

# Required status checks = the CI matrix + the Gitflow guard jobs (by name).
CONTEXTS='["lint-type-test (ubuntu-latest)","lint-type-test (windows-latest)","Branch naming","PR base branch"]'

echo "Protecting main on $REPO (PR required, CI must pass, admins included, no force-push/delete)..."
gh api --method PUT "repos/$REPO/branches/main/protection" \
  -H "Accept: application/vnd.github+json" --input - <<EOF
{
  "required_status_checks": { "strict": true, "contexts": $CONTEXTS },
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 0 },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Protecting develop on $REPO (CI on PRs; admins may push small fixes; no force-push/delete)..."
gh api --method PUT "repos/$REPO/branches/develop/protection" \
  -H "Accept: application/vnd.github+json" --input - <<EOF
{
  "required_status_checks": { "strict": true, "contexts": $CONTEXTS },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Done. Verify in the repo's Settings -> Branches."
