# Canopy review API

Fastify + Prisma (PostgreSQL) + S3 service that owns reviews, the recorded
run, and every photo/video in object storage. Design and endpoint contract:
`docs/adr/0017-intake-review-api-and-run-capture.md`. Nothing in `canopy`
(the Python package) imports or calls this service.

## Run it locally

```bash
# from src/api/
docker compose up -d          # Postgres on 5432, MinIO on 9000/9001
cp .env.example .env          # edit if you changed compose ports/creds
npm install
npm run db:migrate            # applies prisma/migrations, generates the client
npm run dev                   # tsx watch, http://localhost:4000
```

`GET /health` should return `{"ok":true}` once it's up.

The viewer page (`viewer.review_api_url`) and the dashboard
(`NEXT_PUBLIC_CANOPY_API_URL`) both default to `http://localhost:4000` and
talk to this service directly; neither needs the docker-compose file.

## MinIO console

http://localhost:9001, user/pass `canopy` / `canopy12345` (see
`docker-compose.yml`). The `canopy` bucket and its CORS rule (GET/PUT/HEAD
from any origin, see `minio-cors.json`) are created by the one-shot
`minio-init` container on `docker compose up`.

## Environment variables

See `.env.example`. With `S3_ENDPOINT` unset, the S3 client talks to real
AWS S3 instead of local MinIO -- set `S3_REGION`/`S3_BUCKET`/credentials to
a real bucket for that. `ANTHROPIC_API_KEY` is optional: without it,
`PUT /reviews/:id/run` still analyses the run and returns a full report, using
deterministic text instead of an LLM-written one.

## Scripts

| Command | Does |
| --- | --- |
| `npm run dev` | tsx watch |
| `npm start` | run once, no watch |
| `npm run build` | `tsc --noEmit` |
| `npm run db:migrate` | `prisma migrate dev` |
| `npm test` | `node:test` via tsx -- covers the deterministic half of `src/analysis.ts` (blocker mapping, severities, recommendation choice, fallback text). No network calls. |

## Schema changes

Edit `prisma/schema.prisma`, then `npm run db:migrate -- --name <change>`
against the compose Postgres. Commit the generated `prisma/migrations/`
directory.
