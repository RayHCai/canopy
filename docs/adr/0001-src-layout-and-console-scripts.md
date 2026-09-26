# 1. `src/` layout and console-script entry points

Date: 2026-09-25
Status: accepted

## Context

`spec.md` sketches the tree as a top-level `canopy/canopy/` package with CLI
files in `scripts/`. Two problems with taking that literally:

1. With the package directory next to the tests, `import canopy` resolves to the
   source tree whether or not the package is installed. Tests then pass against
   a half-installed project, and packaging bugs (a missing `py.typed`, a data
   file absent from the wheel) surface only for someone else, later.
2. Loose files in `scripts/` are not importable, so they cannot be tested,
   type-checked or reused. CLI argument wiring is exactly the code that benefits
   from a test.

## Decision

Use a `src/canopy/` layout. Register CLIs as console scripts in
`pyproject.toml`, implemented as importable modules under `src/canopy/cli/`.
`scripts/` holds shell bootstrap only.

The module-per-boundary structure inside the package is unchanged from the spec,
so `canopy/sim/sensors.py` and friends land exactly where it says.

## Consequences

- `uv sync` installs the project editable, so imports always go through the
  installed distribution.
- `canopy-fly` is on `PATH` inside the environment; `python scripts/...` is not
  how anything is invoked.
- Each CLI has tests that call `main(argv)` directly and assert its exit code.
