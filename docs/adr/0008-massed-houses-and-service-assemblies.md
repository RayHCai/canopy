# 8. Massed houses, service assemblies, and the rest of the model library

Date: 2026-09-25
Status: accepted

Supersedes the "Footprint", "Stories" and "Roof, doors, windows, garage"
bullets of [ADR 0007](0007-model-library-worldgen.md). The rest of 0007 — the
YAML model library, the material-to-class mapping, the placement-rule registry,
`generate_field` — stands unchanged.

## Context

ADR 0007 took the house from the one authored shell in `assets/obj_export/`,
because that was the only house art available. Measuring it established that all
twelve authored homes share an identical 14.90 x 11.02 x 5.36 m shell, so a
generated property's house never varied in shape and "one story or two" was not
reachable at all. 0007 recorded that as a known limitation.

It was also only using five of the fourteen usable props. `breaker_panel`, the
three conduit fittings, `shed` and `fence_panel` had no semantic class and no
role, so they never appeared. And the three service props are not independent
scenery: a meter, its panel and the conduit between them only mean anything as a
connected assembly, which no per-prop placement rule could express.

The twelve authored homes turned out to be the specification for that assembly.
Measured in the world frame, all twelve put the electric meter's underside at
z = 1.25, and two topologies follow:

| Topology | Homes | Arrangement |
|---|---|---|
| panel | 10 of 12 | breaker panel 0.60-1.05 m along the same wall, underside z = 0.75; conduit nipple bridging at z ~ 1.50 |
| riser | 2 of 12 | no exterior panel; an LB riser ~0.85 m along the wall, spanning z 0.37-1.80, carries the service through the wall |

Gas meters sit at grade (z 0-0.72), on the same wall eight times and a different
one four times.

## Decision

**Houses are a union of blocks.** A house is one to three axis-aligned
`Mass` rectangles — a main block plus abutting wings — unioned with
`shapely.unary_union`. `HouseFrame` is now that union: `footprint` is its
outline, `walls` are the outline's exterior edges with outward normals, and
everything that measures from "the house" measures from those. Interior rings
are dropped; a courtyard is not a plan this generator should produce. Each block
gets its own roof, gable or flat, sized from the pitch and the span.

**Both house styles coexist, chosen per seed.** `p_authored` (0.4) of
properties get the authored brick shell, which brings real windows, doors and a
garage; the rest get procedural massing, which brings genuinely varying
dimensions and L or T plans. One rule, `house_lot`, owns the choice, because a
property has exactly one house.

**The electrical service is one rule, not three props.** `service_assembly`
places the meter, then rolls the topology above, then optionally adds a vertical
run from grade. Every property gets a meter and conduit — that is a hard
requirement, not a probability, because a property without them has no mission.
The pieces stay separate `SceneObject`s so the detector, the coverage metric and
the site solver see a meter, a panel and conduit rather than one lump.

**Material-to-class mapping is overridable per model.** The library-wide map is
keyed on material name alone, and the art reuses names across very different
props. Without an override a shed's `concrete` pad read as `DRIVEWAY` and its
`wood_dark` trim as `FENCE`, both of which would mislead the site solver. A
model's own `materials:` block now wins over the library's.

**Three classes appended to `Cls`**: `PANEL = 13`, `CONDUIT = 14`,
`SHED = 15`, with colours. Append-only, so manifest round-trips are unaffected.

**Sheds and fences.** Sheds (0-2) use `yard_scatter` with a new
`back_yard_only` parameter, so they land behind the house rather than on the
elevation the survey photographs. `fence_perimeter` tiles `fence_panel` around a
lot inset sampled per property, usually leaving the street side open; its
`count` is whether the property is fenced at all, not a panel count.

## Consequences

- All fourteen usable props are now placed. Across twelve seeds: meter 12/12,
  AC 12/12, bushes 12/12, trees 12/12, panel 10/12, nipple 10/12, straight 9/12,
  shed 8/12, fence 5/12, gas meter 5/12, LB riser 2/12. The riser ratio
  reproduces the authored homes' 2-of-12 exactly, which is the check that the
  topology roll is calibrated rather than invented.
- Three assets are deliberately excluded. `canopy_scout` is the drone and
  `base_core_battery` is the SSR *output*, both per instruction;
  `base_power_scene_assets` is a catalog sheet containing copies of all thirteen
  other props laid out in a row, not a placeable object.
- `HouseFrame.size` is now the union's bounding box, not a single block's, and
  `contains` tests each block rather than one rectangle. `house_frame()` takes a
  sequence of `Mass` instead of centre/extents/yaw.
- Object counts roughly doubled on fenced seeds — up to 96, of which 46 are
  fence panels. Triangle counts are unchanged in character (145k-251k), because
  the fence is cheap and the authored vegetation still dominates.
- The procedural houses are untextured boxes. They have no windows, doors or
  garage, so on those seeds the `DOOR` and `WINDOW` classes come only from a
  shed, and the ground-band coverage metric has correspondingly less to find.
  Adding opening panels is a contained follow-up: thin boxes on the wall
  segments the frame already exposes.
  *Update:* done. The `facade_openings` rule puts a front door, windows on
  every storey and sometimes a garage door on procedural houses, skipping the
  authored shell (`HouseFrame.has_openings`). It runs after every other
  surveyed-lot role, so it fits around the wall-mounted service equipment
  (read from the new `PlacementContext.placed`) and leaves every existing
  seed's other objects byte-identical.
- Avoidance is verified end to end rather than assumed. `generate_field` ->
  `sim.scene.load_geometry` -> `SimWorld`/`RaySensor` -> `Mapper` ->
  `ClearanceMap` -> `plan_path` produces paths that clear the real meshes by at
  least 0.60 m of ground truth against a 0.25 m drone radius, across six seeds,
  detouring around the house rather than through it. The fence is dense enough
  that a goal placed outside it is correctly unreachable.
- `worldgen` still owns none of the rendering. The viewer is another session's
  work; this change only guarantees that every object carries a class, a colour,
  a `background` flag and a world-space OBJ.
