# Architecture decision records

Short notes on decisions that depart from `spec.md`, or that a future reader
would otherwise have to reverse-engineer. The spec's own "Key architecture
decisions" section is authoritative and is not duplicated here.

One file per decision, numbered, never edited after acceptance -- supersede
instead.

| ADR | Decision |
|---|---|
| [0001](0001-src-layout-and-console-scripts.md) | `src/` layout and console-script entry points |
| [0002](0002-python-311-uv-and-pyenv.md) | Python 3.11 with uv + pyenv, not conda |
| [0003](0003-physics-runs-in-wsl2.md) | The PyBullet track runs in WSL2 |
| [0004](0004-vendored-fork-via-subtree.md) | gym-pybullet-drones vendored via `git subtree` |
