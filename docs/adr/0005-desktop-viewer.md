# 5. A desktop viewer: pywebview window, three.js scene

Date: 2026-09-25
Status: accepted

## Context

The only way to *watch* a flight was `canopy-fly --dynamics pybullet --gui`,
which opens PyBullet's OpenGL example browser. That window is a debugging tool:
Explorer/Test/Params panels, File/View menus, an example-description pane, a
checkered `plane.urdf` floor, RGB axis lines, aliased rendering and a title
hard-coded in Bullet's C++. It also only exists under WSL (ADR 0003).

We want a clean, modern viewer: a white sandbox with soft shadows, a free camera
(WASD and mouse), and a single settings panel behind a gear icon with animated
transitions -- starting with swarm size. PyBullet's window cannot host custom
UI at all, and Rerun (the spec's viz choice) is an analysis tool whose chrome we
cannot restyle or extend with controls that drive the simulation.

A desktop window was preferred over a browser tab. Rendering speed is not the
reason -- both draw with WebGL on the GPU, and Python is the bottleneck either
way -- but a native window is what users expect from a desktop app.

## Decision

`canopy-view` opens a native window through **pywebview** (Edge WebView2 on
Windows) that loads a static page from `src/canopy/viz/web/`. The page renders
with **three.js r170**, vendored as a single ES module under `web/vendor/` so
the viewer works offline and the version cannot drift.

The split of responsibilities:

- `canopy.viz.viewer.ViewerSession` owns simulation state and time. It is pure
  Python, headless and unit-tested. The page reports elapsed wall-clock time
  each animation frame; the session runs whole fixed control ticks and returns
  poses interpolated between the last two, so a 20 Hz simulation draws smoothly
  at the display rate while its dynamics stay identical to `canopy-fly`.
- The page owns presentation only. It never decides where a drone goes.
- The bridge (`_Bridge`) exposes four calls: `scene`, `frame`,
  `set_drone_count`, `new_scene`.

pywebview lives in a new `viewer` dependency group, on by default, imported
lazily inside `launch()` so a missing group raises `DependencyMissingError`
instead of breaking `import canopy`.

## Consequences

- Viewing runs natively on Windows with kinematic dynamics; WSL is no longer
  needed to watch a flight. The PyBullet GUI remains available, unchanged, for
  physics debugging.
- Rerun stays the spec's tool for recorded, per-entity mission analysis; the
  two viewers serve different jobs and do not share code.
- A small amount of JavaScript and CSS now lives in the repo. It is not linted
  or type-checked by the Python toolchain; keep it presentation-only so the
  tested Python side stays the source of truth.
- "New random scene" currently re-seeds the swarm layout only. When worldgen
  lands, `ViewerSession.new_scene` is the hook that regenerates the property
  and `scene()` is where its meshes reach the page.
