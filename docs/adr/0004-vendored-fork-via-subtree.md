# 4. gym-pybullet-drones vendored via `git subtree`

Date: 2026-09-25
Status: accepted

## Context

`spec.md` says to `git clone` upstream into `third_party/` and `pip install -e`
it. A bare clone inside the repo is an embedded git repository: it does not
travel with a commit, CI has to fetch it separately, and any local patch lives
nowhere.

Patches are needed. Upstream `main` declares `python = "^3.12"` and
`numpy = "^2.5"`, both of which conflict with the PyBullet wheel situation (ADR
0002), and it declares `torch` (~2.5 GB) as an unconditional runtime dependency
though only two example scripts import it.

## Decision

Vendor the fork with `git subtree add --squash` at upstream
`7ebad1ecabd28a7000add2d05f888aa2e837c2cc`. Patch only `pyproject.toml`, marking
each change `CANOPY PATCH` and documenting the set in
`third_party/gym-pybullet-drones/FORK.md`. Wire it in through
`[tool.uv.sources]` as an editable path dependency.

Library code is left untouched, so future `git subtree pull` merges stay clean.

## Consequences

- The fork and the code depending on it move in one commit; CI needs no network
  access to it.
- `uv sync --extra physics` installs it editable, so it can be stepped into.
- ~15 MB added to the repo, most of it upstream documentation GIFs.
- Upstream syncs are an explicit `git subtree pull` that re-applies four
  `pyproject.toml` patches; nothing else can conflict.
- If a real GitHub fork is wanted later, only the `gpd-upstream` remote changes.
