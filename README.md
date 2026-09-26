# Canopy

A simulated drone swarm autonomously maps a procedurally generated house, finds
the electric meter, and outputs a battery placement decision plus an
auto-captured Site Survey Review (SSR) photo packet.

See [spec.md](spec.md) for the full technical specification, module contracts and
milestone plan. This README covers only how to get it running.

**Status: M0 complete; world generation landed.** The repo skeleton, tooling,
data contracts and kinematic dynamics are in place, and a single drone flies.
`canopy.worldgen.generate_field(seed)` now builds a random residential property
-- house, meter, bushes, trees, AC units -- from the model library in
`assets/`, and writes a `SceneManifest` plus one OBJ per object. No sensing yet,
and the viewer does not render the generated field.

---

## Quick start (Windows)

Prerequisites: [pyenv-win](https://github.com/pyenv-win/pyenv-win) and
[uv](https://docs.astral.sh/uv/). Both are already installed on the dev machine.

```powershell
pyenv install 3.11.9        # once; .python-version pins the project to it
uv sync                     # creates .venv and installs everything
uv run canopy-fly           # a drone climbs to 2 m and orbits the pad
uv run canopy-view          # watch the swarm in the Canopy desktop window
```

Expected tail of the output:

```
flight summary
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

## The viewer

`canopy-view` opens the Canopy desktop window: the swarm orbiting the launch
area in a white sandbox with soft shadows.

| Input | Does |
|---|---|
| `W` `A` `S` `D` / arrow keys | move |
| `Q` / `E` | down / up |
| `Shift` | move faster |
| drag | look around |
| middle-drag | pan |
| scroll | push in / pull out |
| gear, top right (`Esc` closes) | settings: swarm size (1-5), spawn pads, flight trails and detected-object outlines on/off, new random scene |

As the swarm maps, each object it classifies is outlined with a thin box: the
electric meter and its conduit in light blue, bushes near the house in light
red. Detection sees only range and colour, never the simulator's object ids;
the classes, their rules and their outline colours are data
(`perception.classes` and `viewer.detection_colors` in `config/default.yaml`).
See [ADR 0011](docs/adr/0011-colour-and-geometry-object-detection.md).

`canopy-view --drones 5 --seed 42` picks the starting swarm and layout;
`--debug` enables the web inspector. The window is pywebview hosting a three.js
page; see [ADR 0005](docs/adr/0005-desktop-viewer.md).

## Everyday commands

`make` works in WSL and Git Bash; `.\tasks.ps1` is the PowerShell equivalent.

| Task | Command |
|---|---|
| Install / update | `make setup` |
| Lint + format check | `make lint` |
| Auto-fix and format | `make format` |
| Type check | `make typecheck` |
| Tests | `make test` |
| Tests with coverage | `make test-cov` |
| Everything CI runs | `make check` |
| Fly a drone | `make fly` |
| Open the viewer | `make view` |

Install the pre-commit hooks once so `make check` failures never reach CI:

```bash
uv run pre-commit install
```

## Generating a property

```python
from canopy.worldgen import generate_field

manifest = generate_field(seed=42)          # writes out/scenes/42/
```

The same seed reproduces a property byte for byte. What gets placed, how many,
and where is data in [`assets/models/index.yaml`](assets/models/index.yaml):
adding a prop is a model entry plus a role tag, and only a genuinely new *kind*
of placement needs a Python rule. See
[ADR 0007](docs/adr/0007-model-library-worldgen.md).


## Layout

```
config/          all tunables (default.yaml) and placement rules (rules.yaml)
assets/          models/index.yaml is the model + generation-rule database;
                 obj_export/ holds the authored meshes it draws from
scripts/         build_props.py authors those meshes (low-poly, in code);
                 `make props` regenerates them
src/canopy/      the monolith: one subpackage per module boundary
  contracts.py   every dataclass that crosses a boundary
  config.py      YAML -> frozen, validated dataclasses
  worldgen/      assets (model library), placement (rules), generate, objio
  sim/           dynamics, and later scene, sensors, world
  planning/      waypoints, and later frontier, assign, pathing, mission, safety
  perception/    object detection from range and colour: detector, colour, cells, boxes
  mapping/ site/ report/ viz/    stubs, filled per milestone
  cli/           console entry points
tests/           mirrors src/canopy
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
uv add --group rl some-pkg           # an optional track
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

## Licence

Canopy is MIT.
