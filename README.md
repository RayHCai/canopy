# Canopy

A simulated drone swarm autonomously maps a procedurally generated house, finds
the electric meter, and outputs a battery placement decision plus an
auto-captured Site Survey Review (SSR) photo packet.

See [spec.md](spec.md) for the full technical specification, module contracts and
milestone plan. This README covers only how to get it running.

**Status: M0 complete.** The repo skeleton, tooling, data contracts and both
dynamics backends are in place, and a single drone flies. No scene, no sensing,
no obstacles yet.

---

## Quick start (Windows)

Prerequisites: [pyenv-win](https://github.com/pyenv-win/pyenv-win) and
[uv](https://docs.astral.sh/uv/). Both are already installed on the dev machine.

```powershell
pyenv install 3.11.9        # once; .python-version pins the project to it
uv sync                     # creates .venv and installs everything
uv run canopy-fly           # a drone climbs to 2 m and orbits the pad
```

Expected tail of the output:

```
flight summary
  dynamics         kinematic
  waypoints        49/49
  completed        yes
  ...
```

`uv run` activates the environment for one command, so there is nothing to
source. To get a persistent shell instead:

```powershell
.\.venv\Scripts\Activate.ps1
canopy-fly --laps 3 --radius 8
```

## Physics mode (WSL2)

PyBullet has never published a Windows wheel, and publishes no CPython wheel
above 3.11. Rather than compile Bullet's C++ with MSVC, Canopy runs the physics
track under WSL2, where the prebuilt Linux wheel installs in seconds. This is
the spec's documented fallback, and it costs nothing: physics is an optional flag
and the critical path is kinematic.

WSL2 on Windows 11 ships WSLg, so the PyBullet GUI opens as a normal window with
no X server to configure.

```bash
wsl -d Ubuntu
cd /mnt/c/Users/<you>/.../canopy
./scripts/bootstrap-wsl.sh              # installs uv + Python 3.11, syncs the physics group
export UV_PROJECT_ENVIRONMENT=.venv-wsl # the Linux venv, kept separate from Windows'
uv run canopy-fly --dynamics pybullet --gui
```

`--gui` throttles the simulation to wall-clock time so the flight is actually
watchable; otherwise it runs ~60x faster than real time and the window just
flashes past. Use `--no-realtime` to opt out, or `--realtime` headless.

| | Windows (native) | WSL2 Ubuntu |
|---|---|---|
| Kinematic dynamics | yes | yes |
| Ray casting, Rerun, world gen | yes | yes |
| PyBullet physics | no wheel | yes, prebuilt |
| Recommended for | day-to-day dev, the live demo | `--dynamics pybullet` |

The two environments are separate `.venv` directories built from the same
`pyproject.toml` and `uv.lock`, so they never disagree about versions.

## Everyday commands

`make` works in WSL and Git Bash; `.\tasks.ps1` is the PowerShell equivalent.

| Task | Command |
|---|---|
| Install / update | `make setup` |
| Environment without physics | `make setup-lean` |
| Lint + format check | `make lint` |
| Auto-fix and format | `make format` |
| Type check | `make typecheck` |
| Tests | `make test` |
| Tests with coverage | `make test-cov` |
| Everything CI runs | `make check` |
| Fly a drone | `make fly` |
| Fly with physics (WSL) | `make fly-physics` |

Install the pre-commit hooks once so `make check` failures never reach CI:

```bash
uv run pre-commit install
```

## Layout

```
config/          all tunables (default.yaml) and placement rules (rules.yaml)
src/canopy/      the monolith: one subpackage per module boundary
  contracts.py   every dataclass that crosses a boundary
  config.py      YAML -> frozen, validated dataclasses
  sim/           dynamics, and later scene, sensors, world
  planning/      waypoints, and later frontier, assign, pathing, mission, safety
  worldgen/ mapping/ perception/ site/ report/ viz/    stubs, filled per milestone
  cli/           console entry points
tests/           mirrors src/canopy
third_party/     vendored gym-pybullet-drones fork (see its FORK.md)
out/             generated scenes, recordings, packets (gitignored)
```

Three deliberate deviations from the layout in `spec.md`, each recorded as an ADR
in [docs/adr/](docs/adr/):

- **`src/` layout** instead of a top-level `canopy/` package, so tests always
  exercise the installed package rather than a shadowing source directory.
- **CLI entry points in `src/canopy/cli/`**, registered as console scripts,
  instead of loose files in `scripts/`. `scripts/` holds shell bootstrap only.
- **uv + pyenv** instead of conda, and **Python 3.11** instead of 3.10.

## Adding a dependency

Never `pip install` into the venv; it will be silently reverted by the next sync.

```bash
uv add shapely                       # runtime
uv add --group dev pytest-benchmark  # tooling
uv add --group physics some-pkg      # an optional track
```

`uv.lock` is committed and is the reproducibility guarantee. `uv sync` makes the
environment match it exactly.

## Gotchas

- **Windows spawns processes instead of forking.** Every entry point needs an
  `if __name__ == "__main__":` guard, and each worker must build its own Open3D
  scene — those objects are not picklable.
- **OneDrive.** This checkout lives under `OneDrive\Desktop`, so OneDrive will
  try to sync `.venv` (~1 GB) and everything written to `out/`. Both are
  gitignored but not sync-excluded. Either exclude them in OneDrive settings, or
  move the checkout somewhere like `C:\dev\canopy`.
- **Rerun on a 16 GB laptop:** launch the viewer as `rerun --memory-limit 4GB`.
- **The physics drone is a Crazyflie 2.X** — a ~27 g indoor quadcopter, and
  `DSLPIDControl` is tuned to match (upstream's own example flies 0.3 m circles).
  Handing it a waypoint metres away saturates the attitude command and throws the
  drone across the lot. `PyBulletDynamics` therefore runs a reference governor:
  the controller only ever sees a setpoint creeping toward the waypoint at
  `physics.setpoint_speed_ms` (default 0.6 m/s). That is why physics mode flies
  at ~0.6 m/s where kinematic flies at `sim.v_max` = 3.0 m/s. Raising the
  governor much above 1.0 m/s destabilises the airframe.

## Licence

Canopy is MIT. The vendored `third_party/gym-pybullet-drones` is MIT and retains
its original licence and attribution.
