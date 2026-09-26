# 11. Object detection from range and colour, not object ids

Date: 2026-09-26
Status: accepted

## Context

`spec.md` gets semantics for free. Every ray returns the id of the object it
hit, and `GTDetector` reports that object's manifest class at confidence 1.0;
only the meter was ever to get a real detector, as a stretch goal. The
swarm therefore never has to recognise anything, and nothing about how it
finds the meter would carry over to a real drone.

The design doc "Canopy: Perception and Lot-Wide Mapping Design" lays out the
replacement: an information boundary that keeps ids away from the algorithm,
colour from the sensors, and a learned segmentation model on RGB-D frames
whose masks vote into the voxel map. This ADR lands a first, lightweight
stage of it: a classical detector on the 360 ranger's range and colour. It
finds the electric meter, the conduit and the bushes near the house, and
boxes each one as tightly as the evidence allows. The learned model stays
future work behind the same output.

## Decision

- **Information boundary.** A new `Observation` contract carries what a drone
  senses: its own position, ray directions, ranges and colours. `Scan` keeps
  its object and triangle ids for the ground-truth coverage metric and the
  reveal, and `Scan.observation()` drops them. `Mapper` hands the detector only
  the observation. `tests/perception/test_boundary.py` fails if the perception
  package ever names `obj_ids`, `tri_ids`, the manifest or the class palette.
- **The ranger reports colour.** Each hit's colour is the object's true colour
  lit by a fixed sun, `ambient + (1 - ambient) * max(0, n . sun)`, plus
  per-channel noise (`sensor.sun_dir`, `ambient`, `rgb_noise`, `sky_rgb`).
  Colour noise draws from a child of the sensor's generator, so every range
  sample and scan jitter, and with them every flown mission, is unchanged.
- **Classes are data.** `perception.classes` in `config/default.yaml` lists one
  entry per class: an HSV colour rule, the height band its evidence is
  collected from, and optional shape rules (extents, elongation, height,
  bottom) and a `near_cls` rule. Adding a class means adding an entry, plus an
  outline colour under `viewer.detection_colors`. A `report: false` class
  (WALL) is context only: "the meter is on a wall", "the bush is near the
  house".
- **Colour rules respect noise.** Hue and saturation survive shading, but a
  pale colour's hue is poorly measured. So a hit or an evidence cell is turned
  away only when its colour is confidently outside the rule: every bound is
  widened by the colour's standard error, which shrinks as hits pool. An
  object's colour, pooled over all its hits, must pass exactly.
- **Evidence, then objects.** Every scan, each coloured hit is pooled into
  the sparse evidence cells of every class it matches. Once a second
  (`extract_hz`), counted cells are grouped by connectivity. A class may link
  cells farther apart vertically than horizontally (`link_up_cells`): a thin
  riser is hit at sparse heights, while the panel beside it must stay apart.
  An L-shaped group is cut into straight parts when that shrinks its boxes by
  `split_saving`. Each part is then judged on its own pooled colour and its
  class's shape rules, gets a minimum-area upright box, and keeps its track id
  from the previous extraction if it is found again within `track_match_m`.
  Pale classes (`neighbour_colour`) also judge each cell's colour pooled with
  its neighbours'.
- **Output.** `DiscoveredObject` is keyed by `track_id` (never a scene object
  id) and carries an `OrientedBox`. `MapState.discovered` is replaced, not
  mutated, at each extraction. The viewer outlines each object with thin,
  depth-tested lines in its class's colour, and a settings toggle hides them.

## Consequences

- On five full missions (seeds 1, 3, 7, 11, 42, 3 drones), with ground truth
  used only to score: meter 5/5 and conduit pieces 9/10 (riser and nipple
  boxed separately), with no false positives for either class. Bushes near the
  house were 44/45, also with no false positives. Box centres fall within 1 cm
  for the meter and 1-5 cm for the rest, with extents within a few centimetres
  plus the 3 cm outline margin. The meter is found 10-50 s into a mission.
- Known misses. Two bushes that touch are one connected group, and splitting
  side-by-side blobs saves no box volume, so they get one box. A conduit
  nipple that runs into the breaker panel can be lost with the panel strip it
  merges with.
- Colour is a strong cue because worldgen colours are flat per class. The
  geometry rules settle the palette's collisions: the yellow street line is
  flat on the ground, the grey panel and AC unit are too thick or stand off
  the wall, and a tree reaches above the bush band. The design doc's
  worldgen colour jitter is what would make colour honest. When it lands,
  the colour ranges widen and more of the weight falls on geometry.
- Cost, measured on 600 real scans: `Mapper.integrate` rose from about 8 ms
  to about 10-12 ms per scan on average. That is about 0.5 ms of per-scan
  pooling plus a 20-30 ms extraction once a second; `sensor.scan` gained well
  under a millisecond for shading.
- Nothing plans on `discovered` yet (the INSPECT phase is not built). The
  site solver and INSPECT should read it rather than the manifest.
- Not changed here: the planner still reads `MapState.surface_seen`, which
  coverage builds from ground-truth triangle normals. Moving that behind the
  boundary belongs to the design doc's lot-wide mapping stage.
