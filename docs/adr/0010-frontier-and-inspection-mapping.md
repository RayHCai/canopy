# 10. Swarm mapping: frontiers plus inspection, exterior-only coverage

Date: 2026-09-25
Status: accepted

## Context

`spec.md` Modules 3–5 describe mapping as voxel-frontier exploration: find
FREE voxels next to UNKNOWN ones, cluster them, and send drones there with a
Hungarian assignment. The mission is complete when ground-band coverage (the
area of WALL, DOOR, METER and BUSH triangles below 2.5 m seen by a
photo-quality ray) reaches 0.90. Coverage counts a triangle only when a ray
hits it directly.

Building it end to end against generated properties surfaced three problems
that the spec's rules could not get past:

1. **Voxel frontiers run dry long before the surfaces are seen.** The sensor
   reaches 12 m, but a photo-quality observation needs under 8 m and under 70°
   incidence. A few scans settle almost every voxel of a 30 × 40 m lot, so
   frontier-only exploration stopped at ~54% ground-band coverage with no
   frontier left to visit.
2. **Most WALL area can never be seen.** The authored house shell carries its
   inner wall faces and interior partitions. 63% of WALL-class area (seed 7)
   lies inside the footprint, so ground-band coverage capped near 40% and the
   0.90 completion rule could never fire.
3. **Direct hits cannot reveal dense meshes.** Authored vegetation has ~9k
   triangles per bush. A triangle revealed only when a ray lands on it leaves
   those meshes speckled grey no matter how long the swarm flies.

The user also asked for background: neighbouring homes around the lot that
the swarm must not map.

## Decision

- **Inspection targets join frontiers in one pool.** `MapState.surface_seen`
  marks voxels a photo-quality ray has landed in. It is built from scans, so
  the planner may read it. An OCC voxel with a FREE face neighbour that is not
  `surface_seen` is mapped-but-unseen surface. These voxels are bucketed into
  `inspect_bucket_m` cubes. Each bucket gets a viewpoint `inspect_range_m`
  out, within `inspect_max_angle_deg` of its estimated normal
  (`planning/frontier.py::find_inspection_targets`). The same Hungarian
  assignment weighs voxel frontiers and inspection targets on one
  distance-minus-gain scale.
- **Coverage is voxel-granular.** A triangle counts as seen when a
  photo-quality ray hits it, or lands in the map voxel holding its centroid.
- **Coverage counts exterior faces only.** `load_geometry` casts nine sky
  probes from just off each side of every triangle once per scene. A face
  from which none escape is sealed inside other geometry
  (`SceneGeometry.tri_exterior`). Such faces are never revealed and never
  enter a coverage denominator. This is a ground-truth metric computation; the
  planner never reads it.
- **Background lives in the manifest.** Neighbouring lots (lawn plus house)
  are generated beside and behind the surveyed lot with
  `SceneObject.background = True`. Rays hit them, but they lie outside the map
  grid, so the swarm never maps them, and coverage ignores them.
- **Mission phases are TAKEOFF → EXPLORE → RETURN → DONE.** The spec's ORBIT,
  INSPECT (meter close-ups) and PHOTOS phases are not built yet. EXPLORE ends
  when no reachable target remains, or at `timeout_s`. A drone at
  `rth_battery` returns alone.
- **The whole lot is mapped by default.** `done_ground_coverage` is now
  optional and ships as `null`. The spec's 0.90 ground-band stop only measures
  walls, doors, meter, panel and bushes below 2.5 m. On houses with little of
  that, it fired after ~10 s with a third of the lot, lawn and roof included,
  still unmapped (seeds 3 and 23, 5 drones). The user chose full mapping over
  the early stop. Setting the key back to 0.90 restores the spec's rule.
- **Launch and landing are explicit exceptions to "known free only".** Each
  pad's column is seeded FREE as operator-surveyed space
  (`MissionController.launch_boxes`), because no scan from the pad looks
  straight up. `Shield.filter(landing=...)` skips the map check, but not the
  drone check, for the vertical drop onto a drone's own pad. That drop is the
  one move that must end inside the ground margin.
- **Standoffs resolve by the shield's own priority.** After `hold_replan_s`
  held by another drone, the higher id gives way. It first re-routes around
  the others, and failing that steps aside off the right-of-way drone's
  remaining path.

## Consequences

- Seed 7, 3 drones: mapping the whole lot reveals 100% of observable area in
  ~63 s of sim time. The 0.90 ground-band stop would have ended at ~50 s with
  90% revealed. Across the seeds tried there were zero ground-truth
  collisions and zero geofence exits.
- Coverage numbers are no longer comparable with the spec's per-triangle
  definition: they are higher for dense meshes and exclude interior faces.
  The 0.90 threshold keeps its meaning of "the observable ground band".
- `load_geometry` now costs ~0.9 s per scene, most of it the one-off sky-probe
  pass (~4M rays).
- ORBIT, INSPECT and PHOTOS remain to be built on top of `MissionController`
  when the site solver and SSR packet land.
