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
| [0003](0003-physics-runs-in-wsl2.md) | The PyBullet track runs in WSL2 — superseded by [0006](0006-kinematic-only-dynamics.md) |
| [0004](0004-vendored-fork-via-subtree.md) | gym-pybullet-drones vendored via `git subtree` — superseded by [0006](0006-kinematic-only-dynamics.md) |
| [0005](0005-desktop-viewer.md) | Desktop viewer: pywebview window, three.js scene |
| [0006](0006-kinematic-only-dynamics.md) | Kinematic-only dynamics; the PyBullet track is removed |
| [0007](0007-model-library-worldgen.md) | Worldgen composes an authored model library, not procedural primitives — house bullets superseded by [0008](0008-massed-houses-and-service-assemblies.md) |
| [0008](0008-massed-houses-and-service-assemblies.md) | Massed houses, service assemblies, and the rest of the model library |
| [0009](0009-package-facades-and-enforced-layering.md) | Stage packages expose facades; layering is enforced by a test |
| [0010](0010-frontier-and-inspection-mapping.md) | Swarm mapping: frontiers plus inspection, exterior-only coverage, whole lot by default |
| [0011](0011-colour-and-geometry-object-detection.md) | Object detection from range and colour, not object ids |
| [0012](0012-battery-siting-rules-and-wall-candidates.md) | Battery siting: registered rules over candidates along walls traced from occupancy — placeholder values superseded by [0013](0013-battery-space-checklist-and-three-way-verdict.md) |
| [0013](0013-battery-space-checklist-and-three-way-verdict.md) | Battery siting encodes the Battery Space checklist, with a pass / manual review / reject verdict |
| [0014](0014-low-poly-props-and-material-colours.md) | Props are authored in code as low-poly art; the viewer paints material colours while sensing stays per class |
| [0015](0015-address-seeded-sites.md) | Address-seeded sites: open map data as evidence, the model library as the prior, frozen snapshots, invariants with repair |
| [0016](0016-drones-see-only-their-sensors.md) | The swarm sees only its sensors: an operator envelope instead of lot bounds, the house inferred, inspection judged from scan normals, ground truth only as a score |
| [0017](0017-intake-review-api-and-run-capture.md) | Intake questionnaire in the viewer; a Fastify/Prisma/S3 review API the page uploads a captured run to; the dashboard reads only the API |
| [0018](0018-retire-the-report-stage.md) | Retire the empty `report` stage; the SSR packet is assembled by the viewer page and review API (ADR 0017) |
