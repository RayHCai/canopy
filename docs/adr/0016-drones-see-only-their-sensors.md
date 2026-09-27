# 16. The swarm sees only what its sensors see

Date: 2026-09-26
Status: accepted

## Context

`spec.md` settled that "each ray returns distance, object ID and triangle ID,
so semantics are free". ADR 0011 already took object ids away from
perception (`Scan.observation()` strips them), but ground truth still reached
the swarm's decisions by three routes:

1. **Lot bounds.** `manifest.lot_bounds` -- the generator's own box, exact by
   construction -- sized the occupancy grid, set the geofence, kept frontiers
   off the neighbours' yards and kept the detector off the neighbours' meters.
   A real crew does not have the lot to the centimetre, aligned to the frame
   the drones fly in.
2. **Surface normals and labels in "seen well enough".** `surface_seen`, which
   drives the inspection targets and so decides when exploration is finished,
   came from `CoverageTracker`: it looked up each hit's true triangle
   (`scan.tri_ids`), took the mesh's normal for the incidence test and dropped
   hits on objects the manifest flags `background`.
3. **Ground-truth coverage as a stopping rule.** `planner.done_ground_coverage`
   compared against the triangle-area score, built from manifest classes. It
   was off by default, but wired.

A simulation whose planner reads the answer key proves nothing about the
real system: remove it and the swarm may stop too early, too late, or on the
neighbour's house.

## Decision

**The swarm is handed only what a real one would have.** That is each
sweep's range, direction and colour, the lidar's ray layout, its own pose,
where the crew set the pads down, and an operator flight envelope. Everything
else it infers.

- **`Observation` is the only thing the mapper integrates.** `Mapper(cfg,
  envelope, launch_xy)` takes no manifest and no geometry;
  `Mapper.integrate(obs: Observation)`. `MissionRun` is where the boundary is
  drawn: it hands the mapper `scan.observation()` and keeps `Scan` for the
  score.
- **An operator envelope replaces the lot.** `safety.envelope_x_m` /
  `envelope_y_m` are offsets from the pads' centroid (configuration, the
  cleared airspace a crew files, not scene data). They size the map and set
  the geofence (still shrunk by `geofence_inset_m`) and the detector's
  extent. The default, 50 x 44 m reaching 40 m in front of the pads, takes in
  a generated lot and its neighbours.
- **The swarm finds the house itself.** `mapping.property.infer_survey_bounds`
  marks plan columns that fill `map.house_band_z_m` (1.5 to 2.8 m -- the band
  `site.outline` already uses to trace walls, above bushes, meters and fences)
  as wall, unless the space under the band has been seen to be mostly FREE --
  a wall stands on the ground, a tree crown on air and a trunk. A connected
  run that spans at least `house_min_wall_m` is a building (a span, not an
  area, which a round crown also has), and the building nearest the pads is
  the house. `MapState.survey_bounds` is its
  plan box plus `map.survey_margin_m` (6 m). Frontiers, inspection targets
  and reported objects are limited to it. It is recomputed as the map grows,
  so a house first seen as one wall pulls the survey round to the others,
  and the mapper only ever widens it: a later pick can add to the survey but
  never pull it off a house already taken in. (A first version counted cells,
  not span, and did not look under the band: a front-yard crown nearer the
  pads than the house replaced it mid-flight, and the mission ended at 19 s
  with the house unflown.)
- **"Seen well enough" is judged from the scans.** `Observation.grid_shape`
  publishes the azimuth-major range-image layout a spinning lidar reports.
  `mapping.surface.estimate_normals` takes each return's normal from central
  differences across its grid neighbours, and `SurfaceTracker` marks the
  voxels of close, square-on returns. A return on a depth edge gets a normal
  nearly perpendicular to its ray and fails the incidence test, so errors
  cost a revisit rather than leaving a surface unphotographed.
- **Ground truth is a score, kept beside the mission.** `CoverageTracker`
  (triangle reveal, `coverage_ground_band`, `coverage_total`) is owned by
  `MissionRun` and read by the viewer and the report, never by the mapper or
  controller. `MapState.tri_seen` is gone. `done_ground_coverage` now
  compares against `planning.frontier.inspected_fraction`, the swarm's own
  share of mapped ground-band surface marked `surface_seen`.

Pose stays true for now. That stands in for RTK-GPS plus IMU positioning;
pose noise and estimation are separate work.

## Consequences

- The simulation now tests inference that used to be assumed: which building
  is the house, where the survey stops, and when a surface has been
  photographed well enough.
- Measured on generated properties (seeds 0-3, 3 drones, identical scenes
  before and after; coverage is the ground-truth score), every mission
  finished and found the meter both ways:

  | Seed | Mission time | Ground-band coverage | Total coverage |
  | --- | --- | --- | --- |
  | 0 | 86 s -> 89 s | 1.000 -> 1.000 | 0.998 -> 0.985 |
  | 1 | 94 s -> 55 s | 1.000 -> 1.000 | 0.983 -> 0.891 |
  | 2 | 118 s -> 91 s | 0.944 -> 0.999 | 0.974 -> 0.892 |
  | 3 | 90 s -> 42 s | 0.992 -> 0.991 | 0.997 -> 0.754 |

  Ground-band coverage, the part the SSR rests on, held. Total coverage fell
  where the lot runs well past the house: seed 3's back yard lies more than
  `survey_margin_m` behind it, so its trees and fences are no longer flown.
  That is the intended trade (the job is the house, not the lot) and is what
  shortened those missions; a wider margin buys it back.
- The map is larger: 200 x 176 x 48 voxels over the envelope, against
  120 x 160 x 48 over a 30 x 40 m lot. Clearance and frontier extraction
  scale with it.
- **Known limits.** The survey region is an axis-aligned box, so a house
  turned against the world axes gets a looser region. A neighbour's building
  that touches the house (a party wall, a joined garage) merges with it.
  Yard objects further than the margin from the house are no longer
  surveyed. An envelope too small for the job clips the house, and nothing
  warns about it yet.
- `spec.md`'s "semantics are free" decision is amended to match.
