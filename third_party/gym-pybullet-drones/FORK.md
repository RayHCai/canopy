# Canopy fork of gym-pybullet-drones

Vendored into the Canopy monolith with `git subtree --squash`, so the fork and
the code that depends on it move in one commit and CI never needs the network.

| | |
|---|---|
| Upstream | <https://github.com/learnsyslab/gym-pybullet-drones> (formerly `utiasDSL`) |
| Vendored at | `7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (upstream `main`, version 2.2.0) |
| Licence | MIT — see [LICENSE](LICENSE); retained unchanged |
| Local version | `2.2.0+canopy.1` |

## What Canopy uses

Only two classes, both long-stable:

- `envs.CtrlAviary` — multi-drone rigid-body aviary, stepped in DIRECT or GUI mode
- `control.DSLPIDControl` — cascaded position/attitude PID

Both are wrapped by `canopy.sim.dynamics.PyBulletDynamics`, which is the *only*
place in Canopy that imports this package. Nothing else may import it: that
single choke point is what keeps the physics track optional.

Per the spec, the Betaflight and Crazyflie firmware SITL paths
(`envs/BetaAviary.py`, `assets/eeprom.bin`) are unused. They are left in place
rather than deleted, to keep future `git subtree pull` merges clean.

## Patch set

All patches live in `pyproject.toml` and are marked `CANOPY PATCH`. No library
code has been modified.

1. **`python = "^3.12"` → `">=3.11,<3.13"`.** PyBullet has never published a
   Windows wheel and publishes no CPython wheel above 3.11, so upstream's floor
   would force a from-source Bullet build on every platform. All 27 library
   modules parse cleanly under the 3.11 grammar; the pin was a packaging choice,
   not a code requirement.
2. **`numpy = "^2.5"` → `">=2.4,<3"`, `scipy = "^1.18"` → `">=1.17,<2"`.** Both
   upstream pins themselves require Python >= 3.12. These are the newest
   releases that support 3.11.
3. **`torch`, `stable-baselines3`, `control` made extras** (`rl`, `mrac`).
   `torch` is ~2.5 GB installed and is imported only by `examples/learn.py` and
   `examples/play.py`; `control` only by `control/MRACControl.py`, which no
   package `__init__` imports. Canopy's `physics` extra needs none of them.
4. **`pytest` moved to a dev group** — it was declared as a runtime dependency.

## Syncing with upstream

```bash
git fetch gpd-upstream main
git subtree pull --prefix=third_party/gym-pybullet-drones gpd-upstream main --squash
```

Resolve conflicts in `pyproject.toml` by re-applying the four patches above.

If you would rather own a real GitHub fork (to push changes back, or to open
upstream PRs), create it and re-point the remote:

```bash
gh repo fork learnsyslab/gym-pybullet-drones --remote=false   # or use the web UI
git remote set-url gpd-upstream https://github.com/<you>/gym-pybullet-drones.git
```

The vendored tree and the patch set stay exactly as they are.
