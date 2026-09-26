#!/usr/bin/env bash
# Bootstrap the physics track inside WSL2 / Linux.
#
# PyBullet publishes no Windows wheel and no CPython wheel above 3.11, so
# `--dynamics pybullet` runs here rather than on native Windows. WSL2 on
# Windows 11 ships WSLg, so the PyBullet GUI opens as an ordinary window.
#
# Safe to re-run.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

info() { printf '\033[36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$1" >&2; }

# The Windows PATH leaks into WSL, so a bare `uv` can resolve to the pyenv-win
# shim, which cannot execute here. Always prefer a native Linux uv.
find_uv() {
  for candidate in "$HOME/.local/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
    [[ -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  # Fall back to PATH, but only if it is genuinely a Linux binary.
  if command -v uv >/dev/null 2>&1; then
    local found
    found="$(command -v uv)"
    [[ "$found" != /mnt/* ]] && { echo "$found"; return 0; }
  fi
  return 1
}

if ! UV="$(find_uv)"; then
  info "Installing uv (no native Linux uv found)"
  curl -fsSL https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
info "Using uv at $UV"
"$UV" --version

# uv manages its own CPython builds, so Python 3.11 needs no apt or pyenv here.
info "Ensuring Python 3.11 is available"
"$UV" python install 3.11

# The venv must not be shared with Windows: the wheels are platform-specific.
# .venv-wsl keeps the two side by side in the same checkout.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-wsl}"
info "Syncing into $UV_PROJECT_ENVIRONMENT (the physics group is on by default)"
"$UV" sync --python 3.11

info "Verifying the physics stack"
"$UV" run python - <<'PY'
import pybullet
import gym_pybullet_drones
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

print(f"  pybullet             {pybullet.getAPIVersion()}")
print(f"  gym-pybullet-drones  {gym_pybullet_drones.__file__}")
print(f"  CtrlAviary           {CtrlAviary.__name__} ok")
print(f"  DSLPIDControl        {DSLPIDControl.__name__} ok")
PY

if [[ -z "${DISPLAY:-}" ]]; then
  warn "DISPLAY is unset, so --gui will fail. On Windows 11, WSLg should set it
         automatically; try 'wsl --shutdown' from PowerShell and reopen the shell."
fi

cat <<'EOF'

Done. Next:

  export UV_PROJECT_ENVIRONMENT=.venv-wsl
  uv run canopy-fly --dynamics pybullet            # headless
  uv run canopy-fly --dynamics pybullet --gui      # PyBullet window via WSLg
  uv run pytest -m physics                         # the physics tests

EOF
