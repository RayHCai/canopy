# 15. Address-seeded sites: open map data as evidence, the model library as the prior

Date: 2026-09-26
Status: accepted

## Context

`generate_field(seed)` builds a random property. The product question behind
the simulation is always about a real one: a customer's address arrives with
the work order, and a survey of *that* house is what gets flown. So the viewer
should let a user type an address (with autocomplete), confirm it is a home,
rebuild the house and its street from real data with the assets we already
have, fill in whatever the data cannot see, and still guarantee the rules every
property obeys -- above all, that there is a meter to find.

The obvious sources were the wrong ones:

- **Google Earth / Photorealistic 3D Tiles.** A fused photogrammetry mesh, so
  house, tree and lawn are one set of triangles and the meter is not an object
  at all. That breaks the spec's ray-casting decision ("semantics are free"),
  and photogrammetry is a listed non-goal. The Map Tiles policy also forbids
  pre-fetching, caching or offline use of the tiles, which rules out the
  byte-identical OBJs the manifest promises, and allows overlaid 3D objects
  only if they are not "extracted, traced, or otherwise derived" from the
  tiles. The general Maps terms add no tracing building outlines, no 3D building
  models and no index of tree locations built from Maps content. The Solar
  API's roof planes fall under the same content rules.
- **Generative world models** (Genie 3, Marble, Cosmos). They produce a
  plausible world, not this house: none claims metric, address-grounded
  reconstruction, and none labels objects by class.

Vector open data fits instead. A mapped footprint *is* a house, so it is
semantic by construction; it is exactly the granularity the model library
composes at; and OpenStreetMap's ODbL allows keeping it on disk, with
share-alike engaging only if the derived data is distributed publicly.

## Decision

**The data is evidence and the model library is the prior; there is no second
builder.** `generate_field(seed, cfg, site=snapshot)` walks the same roles with
the same rules. Parameters the data observes are pinned: the footprint (fitted
as one to three axis-aligned masses), storeys, roof shape, mapped trees, the
neighbours' footprints, the street and the lot size. Everything the data cannot
see -- openings, the meter and its conduit, gas meter, AC, bushes, fence, sheds
-- is sampled by the rules exactly as for a seed-only property. So "new seed"
on an address-built property re-rolls only what was inferred: each seed is one
hypothesis about the parts of the house nobody has observed.

**One network step, frozen into a snapshot.** `canopy.worldgen.geo` is the only
package allowed to touch the network (`tests/test_architecture.py` enforces it).
`suggest_addresses` (see *Address search* below) and `fetch_site` (one
Overpass query) run once per address and produce a `SiteSnapshot`: world-frame geometry (a building
mapped as a multipolygon counts, as its largest outer ring; a shed or garage is
never taken for the house unless nothing else is mapped), the residential
verdict, and a `SourceRecord` per provider with its licence and attribution.
The snapshot's id is a hash of its content minus retrieval times, it is never
edited, and it lives under `out/sites/<id>/` (gitignored: it holds a home
address). Snapshot plus seed gives byte-identical meshes, offline.

**The world is turned so the house's street faces `-y`.** Every placement rule
was written for that layout -- front door on a `-y` wall, launch pad just
inside the `-y` lot line, frontage strips beyond it, sheds behind the house --
so it keeps working unchanged. The house's dominant wall direction is aligned
with the axes, which keeps the quarter-turn constraint that `HouseFrame` and
the site solver rely on. North can then point anywhere, so
`SceneManifest.north_rad` carries it and anything compass-bound must read it.

**Address search: Photon with no account, Geoapify with a key.** Photon is
built for search-as-you-type (Nominatim's public instance forbids
autocomplete) and needs no account, but it only knows houses mapped in
OpenStreetMap, which in the US is often a minority of them: an address the
user types may simply never be offered. Geoapify's autocomplete adds
OpenAddresses and other authoritative address points to OpenStreetMap,
which is what makes the field behave like the address boxes users know. It
was chosen over Google Places, Mapbox and the rest for its terms: results
may be stored indefinitely provided their attribution travels with them,
which a snapshot kept forever needs, and they may be shown alongside any
map. `worldgen.site.geocoder: auto` uses Geoapify when the environment
variable `geoapify_key_env` names is set, and Photon otherwise; the key never
goes in config. A Geoapify match that comes from OpenStreetMap keeps its
`osm:way/<id>` ref, so the fetch still picks that exact building. Every
provider's candidates then pass through the same typed-address rules
(`geo/search.py`): directional abbreviations are spelled out before the
query is sent ("339 N Oak Park" put two Nova Scotia houses above the right
one), a typed house number must be the one matched, and candidates are
re-ranked by how many typed words they contain.

**The residential gate classifies what stands at the address, not the address
text**, which says almost nothing. It is an additive log-odds score over
mapped evidence -- the building type under the pin, shops or offices inside
it, the land use, footprint size, storeys, a "Suite" in the address -- with
weights in `config/default.yaml`, and three bands: accept, ask the user,
reject. A separate envelope check refuses houses the generator cannot
represent (more than two storeys, a footprint over 400 m^2, or a plan three
axis-aligned blocks fit worse than IoU 0.6), naming the value and the setting.

**The lot is inferred.** OpenStreetMap has no parcels, so the lot is the house
plus default yards, pulled in to the midpoint toward each mapped neighbour,
with its front line 5.25 m short of the street centreline (where
`frontage_strip` lays the street) and never less than the front yard the
launch pads need.

**Hard rules are checked as each role is placed, and repaired.** Invariants
register against a placement rule, as placement rules register against a role.
The meter's are: exactly one meter, on an exterior wall that is not shared
with a neighbour, centre 1.3-1.7 m above grade, clear of the wall's ends, and
with working space in front of it. They run at the role boundary rather than
at the end, because bushes are placed to occlude the meter and openings are
fitted around it: a meter moved afterwards would leave both wrong. A violation
redraws the role from a spawned child generator (so no later role's draws
move) down a ladder of relaxed corner clearances; the winning draw is marked
`repaired`, and if nothing passes generation fails naming the rule. The prior
does the preventing and the invariant the guaranteeing: the meter's wall odds
already skip walls that face a lot line nearer than its working space, because
a repair redraws at random and could miss the one good wall by bad luck. Rules
about the mission rather than the house -- whether INSPECT's close-up has room
in front of the meter -- are reported in `SceneManifest.notes` and never
repaired: real meters do sit in narrow side yards, and repairing them away
would quietly bias the simulation toward easy houses.

**Every object says where it came from.** `SceneObject.provenance` is
`observed`, `inferred` or `repaired`, and the manifest's notes record every
approximation (a hipped roof built as a gable, an imperfect footprint fit).
On an address-built property the meter location is always a hypothesis, and
nothing downstream may present it as surveyed.

**Surfaces.** The viewer's Scene section gains Location: Random | Address,
with an autocomplete field, a build job polled by the page (the fetch runs
outside the session's locks, which the per-frame update also takes), a confirm
step when the gate says "ask", and a site card listing what was observed and
what was inferred -- read off each object's provenance -- with the data's
attribution. An address that fetches but cannot be built fails its job and
leaves the session on the scene it was showing. On the command line:
`canopy-site suggest|fetch|show` and `canopy-view --site <id>`.

## Consequences

- Seed-only generation is unchanged byte for byte: every address-mode branch
  is guarded, and the invariants pass on every existing seed, so no repair
  fires there.
- The meter-wall priors (street-side and side-wall weights, pull toward a
  mapped utility pole) are placeholders, flagged in `assets/models/index.yaml`.
  Calibrate them against surveyed homes the way ADR 0008 calibrated the
  service topology against the authored ones.
- The public Photon and Overpass servers are fair-use and best-effort.
  Fetching once per address suits them; a production deployment needs
  self-hosted instances or a paid provider behind the same `Geocoder` and
  feature-source protocols. Google Places or Address Validation, or USPS RDI
  through a reseller, could slot in there too, subject to their terms (Places
  content may not be used "in conjunction with a non-Google map", and its
  coordinates may be cached for 30 days only).
- The Overpass query uses bounding-box filters only. Asking the server which
  landuse area contains the pin (`is_in`) failed half the time on the public
  instance and found nothing when it answered, so landuse polygons come back
  in the same query and the containment test runs locally. A landuse
  polygon that wholly contains the fetched area, or one mapped as a
  multipolygon relation, is not seen; the residential score then lacks that
  one term. A query Overpass stops early (its `runtime error` remark) fails
  the fetch rather than building from partial data.
- Under load the public Overpass server queues a request for a slot and then
  answers 429/504 or not at all. A busy or timed-out attempt is retried with
  a doubling backoff (`overpass_retries`, `overpass_backoff_s`), with a
  socket timeout of its own (`overpass_timeout_s`, above the query's
  `overpass_query_timeout_s`), and the viewer shows each retry, so a slow
  fetch does not look like a hung one.
- Not rebuilt from data: openings, meter, gas meter, AC, bushes, fences, sheds
  and driveways. Roofs other than gabled or flat are built as gables and noted.
  Houses beyond the envelope are refused rather than faked.
- Neighbours are background scenery built from their real footprints at their
  real orientation; they never enter `HouseFrame`, so they are exempt from the
  quarter-turn rule. A wall a neighbour shares is dropped from the house's
  walls, so nothing is mounted, planted or glazed on it.
- The generated lot follows the real house, so the mapper's voxel grid grows
  with it; `worldgen.site.max_lot_m` bounds that.
- Room to grow, each as an optional extra: Overture buildings for coverage and
  heights, USGS 3DEP lidar for true heights, roof pitch and trees, parcel data
  for real lot lines, and residential weights fitted by logistic regression on
  labelled buildings.
