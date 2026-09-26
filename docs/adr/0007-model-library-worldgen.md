# 7. Worldgen composes an authored model library, not procedural primitives

Date: 2026-09-25
Status: accepted

## Context

`spec.md` Module 1 describes world generation as procedural geometry: extrude a
shapely union of 1-3 rectangles for the footprint, raise gable roofs per
rectangle, cut door and window panels, build the meter from a box plus a
cylinder, and jitter an icosphere per bush. Every object is a few lines of
`trimesh.creation`, and every new kind of object is a new function plus a new
placement branch plus a new colour lookup.

Two things changed that.

First, an authored model library now exists under `assets/obj_export/`: twelve
home scenes and fifteen standalone props (electric meter, gas meter, AC
condenser, bush, tree, fence panel, shed, breaker panel, conduit fittings), all
Y-up, in metres, with each model's origin at its ground or wall contact point.
This art is fixed — it is not ours to regenerate — and it is far better than
anything `trimesh.creation` will produce. Measuring it settled several
questions:

- All twelve homes share one identical house shell, 14.90 x 11.02 x 5.36 m on a
  40 x 40 m yard. They differ only in the electrical equipment and vegetation
  arranged around it, so the homes are twelve pre-composed *layouts*, not twelve
  houses.
- Each home's OBJ separates into named objects (`o house`, `o yard`,
  `o electric_meter`, `o bush_*`, `o tree_*`), so the shell can be lifted out of
  a home and re-composed with independently placed props.
- Materials are semantic: `brick`, `roof_shingle`, `window_glass`, `door_red`,
  `garage_door`, `slab`, `lawn`, `fence_wood`, `leaf`, `bark`.
- The shell's bounding box is its *roof*. Gutters and eaves overhang the brick
  by 0.45 m on the long sides and 0.51 m on the short ones.

Second, the requirement is explicitly that adding models and generation rules
stays easy, because more art is coming.

`trimesh` cannot supply the loader this needs. `trimesh.load` groups an OBJ's
faces by material and merges them *across* `o` objects, so the house comes back
fused with the bushes; `split_object=True` shatters the file into one geometry
per material run — 5,297 of them for one home.

## Decision

World generation is a data-driven composition of a model library.

**One YAML index is the database.** `assets/models/index.yaml` declares
`models:` (what exists), `roles:` (what a property is made of, and in what
order), and `materials:` (material name to `Cls`). Adding a prop means adding a
model entry and tagging it into a role. No Python changes.

**A model is either authored art or primitives.** `mesh:` names an OBJ, narrowed
to one `o` group with `object:`, used at its authored size with `scale:` giving
uniform size variation. `parts:` composes primitives in a normalised unit box
and takes explicit `size_x/y/z` extents. Both end up as a world-space mesh
scaled to chosen extents, so a placement rule cannot tell them apart. This is
what lets the generated ground plane exist with no file on disk, and what lets a
placeholder prop be flown against before its art is drawn.

**Material names carry semantics.** One authored mesh becomes several
`SceneObject`s, one per `Cls` its materials map to. The house shell yields
separate wall, roof, window, door, garage-door and slab objects. This is what
keeps the spec's "semantics are free" ray-casting decision true for authored
geometry, and why the art's material naming is load-bearing.

**Placement is a named-rule registry.** `worldgen/placement.py` holds
`lot_plane`, `house_pad`, `wall_mount`, `wall_adjacent`, `foundation_band` and
`yard_scatter` behind `@register_rule("name")`. A role names one. Writing a rule
is the only thing in worldgen that still requires code, and it is required only
for a placement no existing rule covers.

**`generate_field(seed, cfg)` is the entry point**, in `worldgen/generate.py`.
It walks roles in document order, threads one `default_rng(seed)` through them,
bakes each pose into a world-space OBJ it writes itself, and returns a
`SceneManifest`.

### What this supersedes in the spec

- **Footprint.** One axis-aligned rectangle from the authored shell, turned in
  quarter turns only, rather than a shapely union of 1-3 rectangles. Quarter
  turns keep `HouseFrame` axis-aligned, which the site solver's world-axis cost
  map wants anyway. A second shell, or an L-shaped one, is a new model entry and
  (for a non-rectangular footprint) a new house rule.
- **Stories.** The spec's 50/50 one-story / two-story choice is not available
  from a single 5.36 m shell. Size variety comes from `scale: [0.85, 1.15]` plus
  position and quarter-turn yaw.
- **Roof, doors, windows, garage.** Not generated. They are part of the authored
  shell and are separated out by material instead.
- **Procedural meter, bush and tree geometry.** Replaced by the authored props.
  `p_meter_occluded` and `bush_density_per_10m` still drive placement, so the
  behaviour the spec cared about survives.
- **Driveway and fence.** Deferred. The authored `yard` object has both, but
  positioned for the shell at its authored pose, so it only lines up if the
  house is never moved. It ships as the `yard_authored` model, tagged out of the
  default `ground` role; pointing that role's tag at it uses the authored yard
  instead of the generated plane.
- **`TREE_TRUNK_COLOR`.** Unused. A `SceneObject` carries one colour, and the
  authored tree's `bark` and `leaf` materials both map to `Cls.TREE`. Mapping
  `bark` to its own class would need a `Cls` addition.
- **`scripts/gen_scene.py`.** Not added; entry points belong in `canopy.cli`
  (ADR 0001) and nothing needs one yet. `generate_field` is importable.

## Consequences

- Adding a bush shape, a second house, a shed or a fence style is a YAML edit.
  Adding a *place* for things to go is one function with a decorator.
- `worldgen/objio.py` exists because trimesh cannot read an OBJ the way this
  needs. It is a small, tested file-format boundary: `v`/`o`/`usemtl`/`f`, fan
  triangulation of quads, per-object vertex re-indexing, and the Y-up to Z-up
  conversion. That conversion, `(x, y, z) -> (x, -z, y)`, turns a model's
  authored front (`+z` for this library) into world `-y`, which is why
  `face_axis` is declared per model rather than assumed.
- Anything measuring from a wall measures the `WALL`-class bounding box, not the
  model's. Mounting on the model box put the meter 0.45 m off the brick, in mid
  air. `AssetLibrary.scaled_class_bounds` is the fix and
  `tests/worldgen/test_generate.py` guards it.
- OBJ export is hand-rolled in `generate.py` rather than delegated to trimesh,
  at fixed six-decimal precision and with no library version banner, so a
  reseeded run is byte-identical. The spec's determinism guarantee holds:
  verified across three seeds.
- The spec's 250k triangle budget is exceeded on some seeds (208k-282k
  measured over seeds 1, 7 and 42). The authored vegetation dominates — one tree
  is 19,860 triangles and one bush 9,096 — and the art is fixed. The generator
  logs a warning rather than failing, since a heavy scene still ray-casts fine
  against an Open3D BVH; the cost would land on the reveal renderer. Decimating
  vegetation on import is the lever if it matters, and it belongs to whoever
  finds the frame-time problem.
- Two `mypy` accommodations: `untyped_calls_exclude = ["trimesh"]` in
  `pyproject.toml`, because trimesh ships no type information and the
  alternative is a `type: ignore` on every primitive; and `_primitive`'s
  dispatch over `PartKind` is deliberately exhaustive with no `else`, so adding
  a kind without a builder is a type error rather than a runtime branch that can
  never execute.
- `SceneObject` gained `asset_id`, defaulted to `""`. It is the only record of
  which library entry produced a mesh, which is what makes a surprising scene
  traceable.
- The viewer does not yet render generated fields; `canopy-view` still draws a
  bare ground plane and the swarm. Transporting a manifest's meshes to the
  three.js scene is separate work.
