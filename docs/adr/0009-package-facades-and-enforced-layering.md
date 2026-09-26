# 9. Stage packages expose facades; layering is enforced by a test

Date: 2026-09-25
Status: accepted

## Context

The spec lays Canopy out as one package with one subpackage per pipeline
stage — worldgen, sim, perception, mapping, planning, site, report, viz — and
says every module is testable alone. It does not say how stages may depend on
each other, and until now nothing checked.

A survey at the end of M4 found the dependency direction clean (every
cross-stage import pointed from a later stage to an earlier one, no cycles),
but held that way by discipline alone. It also found that only `worldgen`
declared a public API. Every other stage was consumed by reaching into its
submodules — `viz.viewer` imported `canopy.sim.scene`, `canopy.sim.sensors`,
`canopy.planning.run` and three `canopy.worldgen.*` modules directly, even
bypassing the `worldgen` API that did exist. With no line between a stage's
surface and its internals, any helper was one import away from becoming a
cross-stage dependency, and renaming or splitting a module meant auditing the
whole tree.

## Decision

Treat each stage package as a module of a modular monolith:

- **Facade.** Each stage's `__init__.py` re-exports the names other packages
  use and lists them in a sorted `__all__`. That is the stage's public API.
- **Cross-stage imports go through the facade** — `from canopy.sim import
  SimWorld`, never `from canopy.sim.world import SimWorld`.
- **Intra-stage imports stay direct**, and a module never imports its own
  package's facade. The facade imports the submodules; a submodule importing
  the facade back is how import cycles start.
- **Layering.** The core modules (`contracts`, `errors`, `log`, `mathutil`,
  `config`) import only each other. A stage may import core, itself, and
  strictly earlier stages in the order worldgen < sim < perception < mapping <
  planning < site < report < viz < cli.
- **Enforcement.** `tests/test_architecture.py` walks every import in
  `src/canopy` — including `TYPE_CHECKING` and function-local ones — with the
  stdlib `ast` module and fails on any violation. No new dependency
  (import-linter would do the same job but is not worth adding for four rules).

Tests are exempt: a stage's own tests may import its submodules to test them,
and they sit under `tests/<stage>/`, mirroring `src/canopy/<stage>/`.

## Consequences

- Moving, splitting or renaming a submodule is a local change as long as the
  facade keeps exporting the same names.
- A facade grows only when another stage needs a name. Adding to `__all__` is
  the visible, reviewable moment a helper becomes cross-stage API.
- A dependency against the pipeline order — the mapper asking the planner for
  something — fails CI instead of surfacing as an import cycle later. If it is
  genuinely needed, the shared piece moves down into core or an earlier stage.
- Importing a stage runs its whole facade, so `from canopy.sim import
  KinematicDynamics` also imports Open3D. That is acceptable while every
  facade dependency is a core dependency. An optional extra (`rl`, `detect`,
  `viewer`) must still be imported lazily inside the function that needs it,
  never at facade import time.
