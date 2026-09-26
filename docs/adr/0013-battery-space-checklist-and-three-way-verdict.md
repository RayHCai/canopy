# 13. Battery siting encodes the Battery Space checklist, with a three-way verdict

Date: 2026-09-26
Status: accepted

## Context

ADR 0012 made `rules.yaml` a list of registered rules, but its values were
placeholders: `spec.md` listed the real SSR rules as a non-goal. The product
owner has now supplied the real checklist. It is written as a prompt for
reviewing survey photos ("Battery Space"), and several things in it do not
fit the ADR 0012 model:

- **Three outcomes, not two.** A site is `pass`, `reject` or `manual_review`.
  A reject turns a customer away and is expensive. Deferring to a human is
  cheap, so anything uncertain or marginal defers instead of failing: an
  alley near 5 ft, a harness run between 16 and 26 ft, space nobody saw.
- **The harness follows the walls.** It is a cable fastened to the wall every
  ~3 ft, so run length is measured around the house. It may not cross a garage
  door. Running over a door is only occasionally allowed, so that defers.
- **Most checks need objects perception did not report**: doors, garage doors,
  windows, the gas meter, the AC unit and the panel.
- **Bushes are movable**, so they no longer block a site.
- **A street-facing front placement needs the member's sign-off**, so it is
  never an auto-pass.

The prompt also came with checks for other SSR features (meter can size,
CL320, meter number, panel layout, meter-panel relationship). Those are
photo-reading tasks, not placement, and are out of scope here. Where the
supplied sections disagree (the ComEd window-rule and disconnect-placement
variants use 5 ft ground clearances and a separate disconnect box), the
Battery Space section is authoritative. The supplied text was truncated
partway through its decision rules, so the `manual_review` and output
definitions are inferred from the rest of the section.

## Decision

- **Verdicts are data.** `SiteVerdict` (`pass` / `manual_review` / `reject`)
  is a contract type. Every `SiteCandidate` carries one, and `assess_site`
  returns a `SiteAssessment` with the overall verdict and a short
  justification. `suggest_sites` is removed.
- **Rules report a marginal band.** `RuleOutcome.marginal` marks sites that
  are not a clean pass but could be cleared by a measurement or more photos.
  `Rule.required` becomes `on_fail: fail | review | ignore`. A site is
  rejected only by a non-marginal failure of a `fail` rule. The whole survey
  is rejected only when every candidate is rejected, and passes when any one
  passes. One marginal site anywhere is enough to block a reject.
- **Unseen space defers.** `free_space` and `headroom` treat unknown voxels in
  a site's region as marginal, because a site must be seen to be approved.
- **Rule set:**
  - `harness_run` replaces `meter_distance`: along-wall length, ≤ 16 ft
    passes, ≤ 26 ft defers, garage doors block, doors defer.
  - `free_space` measures the alley width: < 4.5 ft fails, 4.5–5.5 ft defers.
  - `headroom` requires 6.5 ft clear above the footprint.
  - `clear_of` gains `plan`, `review_m`, `below_m` and `swing`, which express:
    - 3 ft from the gas meter (safety-critical) and from drivable surfaces;
    - clear of doors and their swing, of garage doors, and of ground-floor
      windows;
    - not under the meter or panel;
    - off the AC unit and the conduit.
  - Bushes become a cost-only preference.
  - `facing` bearing 180 (the street side in worldgen) defers.
- **Perception gains DOOR, GARAGE_DOOR, WINDOW, GAS_METER, AC_UNIT and PANEL**
  under ADR 0011's colour-and-geometry scheme. Doors and garage doors share a
  colour on authored houses, and the AC, panel and conduit share a grey, so
  size and mounting height separate them.
- `spec.md` drops the real-rules non-goal. What remains a non-goal is the
  full photo-review SSR, not the placement checklist.

## Consequences

- **DRIVEWAY has no detector.** The surveyed lot has no driveway. The only
  DRIVEWAY-class geometry is an authored house's foundation-slab band, and
  detecting it would fail every site on its drive clearance. The rule is
  configured and passes everywhere until worldgen generates a real drive.
- **Checks with nothing in the world to measure are not modelled:**
  - permanent pools and their tells, window wells, vents, gates and fence
    doors;
  - decks, porches and slope as ground;
  - main vs secondary walkways, gas pipe runs beyond the meter box;
  - meter accessibility, fixed vs operable windows (every ground-floor window
    is treated as operable);
  - the harness crossing a drivable surface.

  `rules.yaml` lists them. Each becomes a rule when worldgen and perception
  can express it.
- A missed detection now passes a rule it should have failed, such as an
  undetected gas meter inside 3 ft. Detector recall is therefore part of the
  siting's safety, not just its quality.
- The OBJ battery asset is 0.93 m wide because it includes a side disconnect.
  The checklist's battery is 31 in (0.79 m) with no disconnect. Siting uses
  the checklist size; the rendered model is slightly wider than the box that
  was judged.
