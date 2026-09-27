# 17. Intake, the review API, and capturing a run for the dashboard

Date: 2026-09-26
Status: accepted

## Context

The viewer opened straight into a random property and flew it; the dashboard
showed hard-coded members and SVG placeholders. The product flow is one
thing end to end: a member gives their name, email and address, the swarm
surveys *that* house, and a reviewer on the dashboard sees the photos, the
blockers at each candidate battery site, a recommended site, and a recording
of the flight, then emails the member.

That needs somewhere durable for a run to land. The Python package
deliberately has one network boundary (`canopy.worldgen.geo`, ADR 0015) and
no web-service dependencies, and the dashboard is a Next.js app with no
backend of its own.

## Decision

**A separate service, `src/api/`: Fastify + Prisma (PostgreSQL) + S3.** It
owns reviews, the recorded run, and every binary (photos, video) in object
storage. Locally, `src/api/docker-compose.yml` runs Postgres and MinIO, an
S3-compatible store; production points the same client at AWS S3. Nothing in
`canopy` imports or calls it.

**The viewer page, not Python, talks to the API.** The page already holds
the WebGL canvas the photos and the video come from, and a browser can PUT a
blob to a presigned URL without Python growing an HTTP client or an AWS SDK.
Python's part is to *describe* the run: `ViewerSession.run_record()` returns
drone tracks, fleet events, detections, the battery-site assessment and
metrics as plain JSON, and `ViewerSession.intake()` tells the page where the
API is (`viewer.review_api_url`). The network boundary in Python stays where
ADR 0015 put it.

**Intake comes first.** The page opens on a four-step questionnaire (name,
email, address, options) over the idle scene; the mission does not advance
until it is finished. The address step uses the existing ADR 0015 suggest /
build job. **Skip** (lower left) flies the already-loaded random property
locally and uploads nothing.

**Capture.** From the moment the survey starts the page records the canvas
with `MediaRecorder` (WebM). When battery sites first appear, the page
renders one 4:3 "drone photo" per site from a camera standing off the wall,
with the property in true colour and flight overlays hidden, and projects
the battery footprint, the meter and each nearby detection into the photo
as percentage boxes -- the same `Box` shape the dashboard's placeholders
used. When the mission reaches `done` the recording stops and the page
uploads: photos and video to S3 via presigned PUTs, then the run record plus
photo metadata to the API.

**Analysis runs in the API.** For each site the API turns rule warnings and
nearby detections into blockers (type, severity, description, required fix,
box) and recommends a site, with a short reason. A lightweight LLM (Claude
Haiku 4.5) writes the prose from structured facts only -- it never invents a
blocker the facts don't contain, and its output is validated. Without
`ANTHROPIC_API_KEY`, or if the call fails, a deterministic writer produces
the same shape, so a run never gets stuck in "Awaiting Drone Report".

**The dashboard reads only the API.** `mockData.ts` and the placeholder SVGs
go; the queue polls `GET /reviews`, a member page reads `GET /reviews/:id`,
and sending an email records it with `POST /reviews/:id/email`.

## Contract

Base URL `viewer.review_api_url` / `NEXT_PUBLIC_CANOPY_API_URL`, default
`http://localhost:4000`. JSON, CORS open. Times in the run record are
simulated seconds from launch; the API turns them into clock strings from
`startedAt`.

| Method & path | Body | Returns |
| --- | --- | --- |
| `POST /reviews` | `{name, email, address, lat?, lon?, siteId?, droneCount, seed}` | `{id}`; status `Awaiting Drone Report` |
| `POST /reviews/:id/uploads` | `{files: [{name, contentType}]}` | `{uploads: [{name, key, url}]}` -- presigned PUT, 15 min |
| `PUT /reviews/:id/run` | `{record, photos, videoKey}` (below) | the review; status `Ready for Review` once analysed |
| `GET /reviews` | -- | `Member[]` (dashboard `lib/types.ts`) |
| `GET /reviews/:id` | -- | `Member` with `report`, `run`, `sent` |
| `POST /reviews/:id/email` | `{subject, approved, placementPhotoId, blockerPhotoIds}` | the review; status `Email Sent` |
| `GET /media/*key` | -- | 302 to a presigned GET |
| `GET /health` | -- | `{ok: true}` |

`record` is `ViewerSession.run_record()` verbatim. `photos` is one entry per
suggested site: `{siteRank, key, width, height, placementBoxes: [{label,
box}], candidates: [{trackId, cls, box, distanceM}]}`, where `box` is
`{x, y, width, height}` in percent of the image and `candidates` are the
detections within 3 m of the site that the photo can see.

## Consequences

- Two more processes to run for the full flow (`docker compose up`, the API),
  but a viewer with no API reachable still flies, and says it will not upload.
- The recording is whatever the viewer's camera showed, at the playback
  speed chosen; it is a record of the session, not a re-render.
- Photos are rendered, not sensed: they show the property in true colour even
  where no ray landed. They illustrate a site the swarm *did* find, with
  boxes from what the swarm detected, so nothing on them is ground truth the
  swarm lacked -- but they are prettier than the map.
- An address-built meter is still a hypothesis (ADR 0015), and the dashboard
  must not present the review as a site visit.
- LLM text is advisory prose over rule output; the verdicts stay the site
  stage's.
