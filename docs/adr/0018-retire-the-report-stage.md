# 18. Retire the `report` stage

Date: 2026-09-27
Status: accepted

## Context

`spec.md` planned a Python `report` stage after `site`: `photos.py` to compute
an SSR shot list and fly it in a PHOTOS mission phase, and `packet.py` to
write `packet.json`, a matplotlib site plan and a Jinja2 `index.html` under
`out/runs/<run>/packet/`. The package was created as a placeholder
(`canopy/report/__init__.py`, a docstring and nothing else) and never grew.

ADR 0017 then put every piece of that job somewhere else. The viewer page
renders one drone photo per battery site from the WebGL canvas and records
the flight; `ViewerSession.run_record()` describes the run as plain JSON; the
review API stores it, turns rule output into blockers and a recommendation,
and the dashboard is the packet a reviewer reads. The mission has no PHOTOS
phase (ADR 0010's phases are TAKEOFF, EXPLORE, RETURN, DONE).

An empty package that the architecture test lists as a stage suggests a
pipeline step that does not exist, and invites the next change to land in it
rather than where ADR 0017 put the work.

## Decision

Delete `src/canopy/report/` and drop `report` from the stage order in
`tests/test_architecture.py`. The pipeline is
`worldgen -> sim -> perception -> mapping -> planning -> site -> viz -> cli`.

The Python package's product output is the run record plus the site stage's
assessment; the SSR packet is assembled downstream of Python, per ADR 0017.

## Consequences

- `spec.md`'s `report/` layout, the PHOTOS state and the "SSR packet"
  subsection of Module 8 are superseded by ADR 0017 and this record.
- A headless run (`canopy-fly`, batch tests) produces no packet. If an
  offline packet is ever wanted again, it belongs as a consumer of
  `run_record()` (a CLI or a later stage), not as a stage between `site` and
  `viz`.
