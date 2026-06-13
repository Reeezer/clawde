#!/usr/bin/env bash
# clawde one-shot installer (macOS / Linux). Run once after cloning.
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
step() { printf "\n${CYAN}==> %s${NC}\n" "$*"; }
ok()   { printf "    ${GREEN}[OK]${NC}   %s\n" "$*"; }
warn() { printf "    ${YELLOW}[WARN]${NC} %s\n" "$*"; }
fail() { printf "\n    ${RED}[FAIL]${NC} %s\n\n" "$*"; exit 1; }

printf "\n  ${BOLD}clawde -- Setup${NC}\n  ===============\n"

step "Checking uv"
if ! command -v uv &>/dev/null; then
    warn "uv not found -- installing..."
    if command -v curl &>/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then wget -qO- https://astral.sh/uv/install.sh | sh
    else fail "curl or wget required to install uv."; fi
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || fail "uv installed but not on PATH. Open a new terminal and re-run."
fi
ok "uv $(uv --version)"

step "Checking Python 3.12"
uv python find "3.12" &>/dev/null || { warn "Installing Python 3.12 via uv..."; uv python install "3.12"; }
ok "Python 3.12 ready"

step "Syncing the workspace (uv sync --all-packages)"
uv sync --all-packages
ok "Python packages installed"

step "Installing the pre-commit hook"
if uv run pre-commit install >/dev/null; then ok "pre-commit hook installed"
else warn "pre-commit install failed -- run 'uv run pre-commit install' manually"; fi

step "Configuring environment"
if [[ ! -f .env ]]; then cp .env.example .env; ok ".env created from .env.example -- add a provider key (BYOM)"
else ok ".env already exists"; fi

printf "\n  ===============\n  ${GREEN}${BOLD}Setup complete!${NC}\n\n"
printf "  Try it:\n"
printf "    ${CYAN}uv run clawde version${NC}\n"
printf "    ${CYAN}uv run --directory packages/core pytest${NC}\n\n"
