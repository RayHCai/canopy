# 12. Battery siting: registered rules over candidates along traced walls

Date: 2026-09-26
Status: accepted

## Context

`spec.md` Module 7 sites the battery with a 0.1 m 2D cost map (house, wall
band, keep-outs, obstacles, bushes), walks the perimeter from "the convex
outline of occupied wall voxels", and scores with one fixed formula whose
weights live in `rules.yaml`. Three things have changed since it was written:

- Perception (ADR 0011) reports a meter, conduit and bushes, but not walls,
  doors, windows, the AC unit, the gas meter or the breaker panel. Most of
  the spec's keep-out layers have nothing to draw from.
- Houses are unions of blocks (ADR 0008), often L- or T-shaped, so a convex
  outline puts candidate sites in mid-air across the concave corner.
- The product owner wants the placement rules to be as customizable as
  possible: new constraints and factors added over time, not a fixed formula.
  The MVP rules are: on the house's edge, within 6 ft (1.83 m) of the meter,
  and away from bushes. When no site meets them, still show the best option,
  flagged with its issues.

## Decision

- **Walls are traced from occupancy, not assumed convex.** A plan column is
  wall when most of its voxels in a band above head height (1.5-2.8 m) are
  occupied. Bushes, meters and fences stay below that band. The component
  nearest the meter is closed, hole-filled and simplified into a polygon
  whose edges are the walls (`canopy.site.outline`). On seeds 42, 7 and 123
  this matches the true footprint to IoU 0.93-0.97. The manifest footprint is
  never read.
- **The search space is candidates along the walls, not a raster.** A battery
  hangs on a wall, so the only freedom is where along which wall. Candidates
  sit every 0.1 m on each wall long enough for the battery, so "along the edge
  of the house" holds by construction. Clearances are exact box-to-box gaps
  (`canopy.site.geometry`), not cost-map lookups.
- **`rules.yaml` is a list of registered rules.** Each entry names a `Rule`
  subclass and gives its parameters, which are type-checked with the same
  strictness as `default.yaml` (`canopy.config.build_section`). Every rule is
  both a constraint (`required`, default on) and a cost factor (`weight`).
  New rules are added with `@register_rule`, with no solver change. The
  built-in rules are `meter_distance`, `clear_of <class>`, `free_space`
  (occupancy in the battery's footprint plus working clearance, which catches
  obstacles perception has no class for) and `facing`.
- **Rule-breaking sites are ranked, not dropped.** Sites that pass every
  required rule rank first by cost. Flagged sites follow, fewest broken rules
  first, and are offered when too few pass. `SiteCandidate` gains a
  `warnings` tuple, one sentence per broken rule. The viewer outlines those
  sites in orange instead of yellow.

## Consequences

- Things mounted flush on a wall are invisible to siting. At 0.25 m voxels a
  breaker panel is indistinguishable from the wall behind it, and perception
  has no `PANEL` class, so a suggested site can sit under the panel. Adding
  a `PANEL` detector class makes a `clear_of: PANEL` rule work without further
  changes.
- The spec's `bushes_to_remove` stays empty. The MVP rule avoids bushes
  instead of pricing their removal. A removal rule would be a new `Rule` whose
  cost counts the bushes in the way.
- `rules.yaml` has a new shape, so anything reading the old flat keys
  (`battery_w_m`, `weights`) must move to `canopy.site.load_site_rules`.
- Siting runs once, when the mission leaves exploration, on the viewer's
  simulation thread. It takes a few hundred milliseconds, well inside the
  playback lead.
