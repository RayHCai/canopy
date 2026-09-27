/** Review lifecycle: create, list, fetch, submit a run, and send the email. */
import type { FastifyInstance } from "fastify";
import type { Prisma } from "@prisma/client";
import { analyzeRun } from "../analysis.js";
import { prisma } from "../db.js";
import { NotFoundError } from "../errors.js";
import { REVIEW_INCLUDE, toMember } from "../mapping.js";
import { createReviewSchema, emailRequestSchema, runRequestSchema, type RunPhoto } from "../schemas.js";
import type { SentEmail } from "../dashboardTypes.js";

export async function reviewRoutes(app: FastifyInstance): Promise<void> {
  app.post("/reviews", async (req, reply) => {
    const body = createReviewSchema.parse(req.body);
    const review = await prisma.review.create({
      data: {
        name: body.name,
        email: body.email,
        address: body.address,
        lat: body.lat ?? null,
        lon: body.lon ?? null,
        siteId: body.siteId ?? null,
        droneCount: body.droneCount,
        seed: body.seed,
        status: "AWAITING",
      },
    });
    reply.code(201);
    return { id: review.id };
  });

  app.get("/reviews", async () => {
    const reviews = await prisma.review.findMany({
      include: REVIEW_INCLUDE,
      orderBy: { createdAt: "desc" },
    });
    return reviews.map(toMember);
  });

  app.get<{ Params: { id: string } }>("/reviews/:id", async (req) => {
    const review = await prisma.review.findUnique({ where: { id: req.params.id }, include: REVIEW_INCLUDE });
    if (!review) throw new NotFoundError(`review not found: ${req.params.id}`);
    return toMember(review);
  });

  app.put<{ Params: { id: string } }>("/reviews/:id/run", async (req) => {
    const { id } = req.params;
    const existing = await prisma.review.findUnique({ where: { id } });
    if (!existing) throw new NotFoundError(`review not found: ${id}`);

    const body = runRequestSchema.parse(req.body);
    const { record, photos, videoKey } = body;

    const photosByRank = new Map<number, RunPhoto>(photos.map((p) => [p.siteRank, p]));
    const sites = record.site?.sites ?? [];
    const analysis = await analyzeRun(sites, photosByRank);

    const recommendation =
      sites.length > 0
        ? { placementPhotoId: `plc-${analysis.recommendedRank}`, reason: analysis.recommendationReason }
        : null;

    const sitesData: Prisma.SiteSuggestionCreateWithoutReviewInput[] = sites.map((site) => {
      const photo = photosByRank.get(site.rank);
      const siteAnalysis = analysis.sites.find((s) => s.rank === site.rank);
      return {
        rank: site.rank,
        pos: site.pos as unknown as Prisma.InputJsonValue,
        meter: site.meter as unknown as Prisma.InputJsonValue,
        yaw: site.yaw,
        cost: site.cost,
        verdict: site.verdict,
        breakdown: site.breakdown as unknown as Prisma.InputJsonValue,
        warnings: site.warnings as unknown as Prisma.InputJsonValue,
        photoKey: photo?.key ?? null,
        photoW: photo?.width ?? null,
        photoH: photo?.height ?? null,
        placementBoxes: (photo?.placementBoxes as unknown as Prisma.InputJsonValue) ?? undefined,
        summary: siteAnalysis?.summary ?? null,
        blockers: {
          create: (siteAnalysis?.blockers ?? []).map((b) => ({
            type: b.type,
            severity: b.severity,
            description: b.description,
            requiredFix: b.requiredFix,
            box: b.box as unknown as Prisma.InputJsonValue,
            trackId: b.trackId,
          })),
        },
      };
    });

    await prisma.$transaction([
      prisma.droneTrack.deleteMany({ where: { reviewId: id } }),
      prisma.fleetEvent.deleteMany({ where: { reviewId: id } }),
      prisma.detection.deleteMany({ where: { reviewId: id } }),
      prisma.siteSuggestion.deleteMany({ where: { reviewId: id } }),
      prisma.review.update({
        where: { id },
        data: {
          status: "READY",
          startedAt: new Date(),
          locationMode: record.location.mode,
          simDurationS: record.sim_time_s,
          mappedAtS: record.mapped_at_s,
          coverageTotal: record.coverage.total,
          coverageGround: record.coverage.ground_band,
          verdict: record.site?.verdict ?? null,
          justification: record.site?.justification ?? null,
          videoKey: videoKey ?? null,
          recommendation: recommendation as unknown as Prisma.InputJsonValue,
          tracks: {
            create: record.tracks.map((t) => ({
              droneId: t.id,
              samples: t.samples as unknown as Prisma.InputJsonValue,
              alive: t.alive,
              battery: t.battery,
            })),
          },
          events: {
            create: record.events.map((e) => ({
              t: e.t,
              droneId: e.drone_id,
              type: e.type,
              message: e.message,
              status: e.status,
              batteryPct: e.battery_pct,
              task: e.task,
            })),
          },
          detections: {
            create: record.detections.map((d) => ({
              detId: d.id,
              cls: d.cls,
              color: d.color as unknown as Prisma.InputJsonValue,
              center: d.center as unknown as Prisma.InputJsonValue,
              size: d.size as unknown as Prisma.InputJsonValue,
              yaw: d.yaw,
              confidence: d.confidence,
            })),
          },
          sites: { create: sitesData },
        },
      }),
    ]);

    const full = await prisma.review.findUniqueOrThrow({ where: { id }, include: REVIEW_INCLUDE });
    return toMember(full);
  });

  app.post<{ Params: { id: string } }>("/reviews/:id/email", async (req) => {
    const { id } = req.params;
    const existing = await prisma.review.findUnique({ where: { id } });
    if (!existing) throw new NotFoundError(`review not found: ${id}`);

    const body = emailRequestSchema.parse(req.body);
    const sentEmail: SentEmail = {
      sentAt: new Date().toISOString(),
      subject: body.subject,
      approved: body.approved,
      placementPhotoId: body.placementPhotoId,
      blockerPhotoIds: body.blockerPhotoIds,
    };

    await prisma.review.update({
      where: { id },
      data: { status: "EMAIL_SENT", sentEmail: sentEmail as unknown as Prisma.InputJsonValue },
    });

    const full = await prisma.review.findUniqueOrThrow({ where: { id }, include: REVIEW_INCLUDE });
    return toMember(full);
  });
}
