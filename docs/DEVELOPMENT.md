# Canopy

A simulated drone swarm maps a procedurally generated house, finds the electric
meter, and outputs a battery placement decision plus a Site Survey Review (SSR)
photo packet. Full spec: [spec.md](../spec.md).

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # install
uv run canopy-view    # desktop viewer: watch the swarm map a house
uv run canopy-fly     # headless single-drone flight
uv run pytest         # tests
```

Dashboard (optional):

```bash
cd src/dashboard && npm install && npm run dev
```

## Reproducing the demo

```bash
uv run canopy-view --drones 5 --seed 42          # same seed -> same property
uv run canopy-site fetch "12 Oak St, Springfield" # real address; prints <site_id>
uv run canopy-view --site <site_id>
```

No keys are required. Optional `.env`:

```bash
GEOAPIFY_API_KEY=            # better address autocomplete (free tier); falls back to Photon
NEXT_PUBLIC_CANOPY_API_URL=http://localhost:4000   # dashboard -> review API
RESEND_API_KEY=              # dashboard email sending
```

## Data

- **Synthetic properties**: generated from a seed by `canopy.worldgen` using
  low-poly meshes authored in code (`scripts/build_props.py`) and placement
  rules in `assets/models/index.yaml`. No external datasets.
- **Real addresses**: footprints, trees and streets from OpenStreetMap
  (© OpenStreetMap contributors, ODbL) via Photon/Overpass, snapshotted to
  `out/sites/`. Meter, bushes and openings are still synthetic.

## Tech stack & architecture

Python 3.11 · NumPy/SciPy · Open3D · trimesh/shapely · pywebview + three.js ·
Next.js dashboard.

```
worldgen -> sim -> perception -> mapping -> planning -> site -> viz (canopy-view)
(house)    (drones,  (detect     (occupancy (frontiers, (battery   |  run record,
            sensors)  meter etc)  + coverage) paths)     siting)    |  photos, video
                                                                    v
                                              review API (src/api) -> dashboard
```

Cross-module types live in `src/canopy/contracts.py`; decisions in
[docs/adr/](adr/).

## Known limitations & next steps

- Kinematic dynamics only — no aerodynamics or wind.
- Detection is rule-based (range + colour), not a learned model; the `detect`
  (YOLO) and `rl` tracks are unfinished.
- Viewer runs are not seed-reproducible end to end.
- Address sites need the house to be mapped in OSM; Overpass is rate-limited.
- Next: learned detection, sim-to-real validation, dashboard wired to live runs.
