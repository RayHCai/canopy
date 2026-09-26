# 3. The PyBullet track runs in WSL2

Date: 2026-09-25
Status: accepted

## Context

PyBullet has never published a Windows wheel, for any version. Native Windows
therefore means compiling Bullet's C++ with MSVC: slow, and a known-fragile
build. `spec.md` sidesteps this with conda-forge, which is not available here
(see ADR 0002), and separately names WSL2 Ubuntu as the fallback.

The architecture makes this cheap. The spec's own decision is that "default
motion model is kinematic, PyBullet physics is an optional flag", and Canopy
touches PyBullet in exactly one module, `canopy.sim.dynamics`.

WSL2 on this machine is Ubuntu 24.04 with WSLg already active (`DISPLAY=:0`), so
the PyBullet GUI opens as a normal window with no X server to set up.

## Decision

Native Windows runs everything except physics. `--dynamics pybullet` runs under
WSL2, bootstrapped by `scripts/bootstrap-wsl.sh` into a separate
`.venv-wsl` from the same `pyproject.toml` and `uv.lock`.

The dependency graph encodes this. The `physics` dependency group carries the
marker `sys_platform != 'win32'`, so a sync on Windows cannot even try to build
Bullet. Because it is a no-op there, the group is listed in
`tool.uv.default-groups`: WSL and Linux get PyBullet with a plain `uv sync`, and
a bare `uv run` never silently strips it back out.

## Consequences

- Day-to-day development, the Rerun viewer and the live demo stay on native
  Windows, where they are fastest.
- The physics tests are marked `physics` and skip themselves via
  `pytest.importorskip`, so the Windows suite stays green.
- CI runs the physics job on `ubuntu-latest` and the rest on both platforms.
- Two environments in one checkout. They share a lockfile, so they cannot
  disagree about versions.
- Optional tracks are PEP 735 dependency groups rather than extras, because uv
  can default-enable a group and (as of 0.11) cannot default-enable an extra.
