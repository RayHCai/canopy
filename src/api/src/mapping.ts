/**
 * Builds the dashboard's `Member`/`DroneReport` shape from stored rows.
 *
 * This is the one place that owns the ADR 0017 response mapping (photo ids,
 * clock-string events, drone final state, ...); routes never assemble these
 * shapes by hand so the contract stays in one spot.
 */
import type { Prisma } from "@prisma/client";
import { env } from "./env.js";
import type {
  Blocker,
  BlockerPhoto,
  Drone,
  DroneReport,
  DroneStatus,
  FleetEvent,
  FleetEventType,
  Member,
  PlacementOverlay,
  PlacementPhoto,
  ReportStatus,
  RunSummary,
  SiteVerdict,
} from "./dashboardTypes.js";

const reviewWithRelations = {
  include: {
    tracks: true,
    events: true,
    detections: true,
    sites: { include: { blockers: true }, orderBy: { rank: "asc" } },
  },
} satisfies Prisma.ReviewDefaultArgs;

export type ReviewWithRelations = Prisma.ReviewGetPayload<typeof reviewWithRelations>;
export const REVIEW_INCLUDE = reviewWithRelations.include;

function reportStatusOf(status: string): ReportStatus {
  switch (status) {
    case "AWAITING":
      return "Awaiting Drone Report";
    case "READY":
      return "Ready for Review";
    case "EMAIL_SENT":
      return "Email Sent";
    default:
      throw new Error(`unknown review status: ${status}`);
  }
}

function mediaUrl(key: string): string {
  return `${env.PUBLIC_URL}/media/${key}`;
}

/** HH:MM:SS in the server's local time zone -- times in the record are simulated seconds from launch. */
function clockString(startedAt: Date, offsetSeconds: number): string {
  const at = new Date(startedAt.getTime() + offsetSeconds * 1000);
  const pad = (n: number): string => n.toString().padStart(2, "0");
  return `${pad(at.getHours())}:${pad(at.getMinutes())}:${pad(at.getSeconds())}`;
}

function distanceToMeterFt(pos: unknown, meter: unknown): number {
  const p = pos as [number, number, number];
  const m = meter as [number, number, number];
  const dx = p[0] - m[0];
  const dy = p[1] - m[1];
  const meters = Math.hypot(dx, dy);
  return Math.round(meters * 3.28084 * 10) / 10;
}

function mapDrones(tracks: ReviewWithRelations["tracks"]): Drone[] {
  return tracks
    .slice()
    .sort((a, b) => a.droneId - b.droneId)
    .map((track) => ({
      id: `Drone-${track.droneId + 1}`,
      status: (track.alive ? "Idle" : "Failed") satisfies DroneStatus,
      batteryPct: Math.round(track.battery * 100),
      currentTask: track.alive ? "Mission complete" : "Signal lost",
    }));
}

function mapEvents(events: ReviewWithRelations["events"], startedAt: Date): FleetEvent[] {
  return events
    .slice()
    .sort((a, b) => a.t - b.t)
    .map((e) => ({
      timestamp: clockString(startedAt, e.t),
      droneId: `Drone-${e.droneId + 1}`,
      type: e.type as FleetEventType,
      message: e.message,
      status: e.status as DroneStatus,
      batteryPct: e.batteryPct,
      currentTask: e.task,
    }));
}

function mapPlacementBoxes(raw: Prisma.JsonValue): PlacementOverlay[] {
  return (raw as unknown as PlacementOverlay[] | null) ?? [];
}

function mapBlockers(blockers: ReviewWithRelations["sites"][number]["blockers"]): Blocker[] {
  return blockers.map((b) => ({
    id: b.id,
    type: b.type as Blocker["type"],
    severity: b.severity as Blocker["severity"],
    description: b.description,
    requiredFix: b.requiredFix,
    box: b.box as unknown as Blocker["box"],
  }));
}

/** Battery sites a reviewer chooses between. Matches `placement.top_k` in config/rules.yaml. */
const MAX_SITES_OFFERED = 2;

/**
 * The best-ranked sites, always including the recommended one. Runs recorded
 * before `top_k` dropped to 2 still carry a third site; this keeps them to
 * the same two options new runs offer.
 */
function offeredSites(
  sites: ReviewWithRelations["sites"],
  recommendedRank: number | null,
): ReviewWithRelations["sites"] {
  const ranked = sites.filter((s) => s.photoKey).sort((a, b) => a.rank - b.rank);
  const recommended = ranked.find((s) => s.rank === recommendedRank);
  const rest = ranked.filter((s) => s !== recommended);
  return (recommended ? [recommended, ...rest] : rest)
    .slice(0, MAX_SITES_OFFERED)
    .sort((a, b) => a.rank - b.rank);
}

function mapPlacementPhotos(sites: ReviewWithRelations["sites"], recommendedRank: number | null): PlacementPhoto[] {
  return sites
    .filter((s) => s.photoKey)
    .map((s) => ({
      id: `plc-${s.rank}`,
      imageUrl: mediaUrl(s.photoKey as string),
      label: `Site ${s.rank}`,
      distanceToMeterFt: distanceToMeterFt(s.pos, s.meter),
      clearancePass: s.verdict === "pass",
      placementBoxes: mapPlacementBoxes(s.placementBoxes),
      verdict: s.verdict as SiteVerdict,
      recommended: s.rank === recommendedRank,
      summary: s.summary ?? `Site ${s.rank}`,
      siteRank: s.rank,
    }));
}

const severityOrder: Record<string, number> = { High: 0, Medium: 1, Low: 2 };

/** A reviewer acts on the worst of what's at a site; the long tail is noise. */
const MAX_BLOCKERS_PER_SITE = 2;

/**
 * One photo per blocker, each highlighting only that blocker, and at most
 * `MAX_BLOCKERS_PER_SITE` per site, most severe first. The swarm takes one
 * photo per site, so a site's blockers share its image; the dashboard shows
 * only the blockers at the site the battery is going to.
 */
function mapBlockerPhotos(sites: ReviewWithRelations["sites"]): BlockerPhoto[] {
  return sites
    .filter((s) => s.photoKey)
    .flatMap((s) =>
      mapBlockers(s.blockers)
        .sort((a, b) => (severityOrder[a.severity] ?? 3) - (severityOrder[b.severity] ?? 3))
        .slice(0, MAX_BLOCKERS_PER_SITE)
        .map((blocker) => ({
          id: `blk-${blocker.id}`,
          imageUrl: mediaUrl(s.photoKey as string),
          blockers: [blocker],
          siteRank: s.rank,
        })),
    );
}

/** Builds the full `Member` shape for a review, including its `report` once the run has landed. */
export function toMember(review: ReviewWithRelations): Member {
  const reportStatus = reportStatusOf(review.status);
  const recommendation = review.recommendation as { placementPhotoId: string; reason: string } | null;
  const sentEmail = review.sentEmail as Member["sent"];

  if (reportStatus === "Awaiting Drone Report" || !review.startedAt) {
    return {
      id: review.id,
      name: review.name,
      address: review.address,
      email: review.email,
      reportStatus,
      createdAt: review.createdAt.toISOString(),
      report: null,
      sent: sentEmail,
    };
  }

  const startedAt = review.startedAt;
  const recommendedRank = recommendation
    ? Number.parseInt(recommendation.placementPhotoId.replace("plc-", ""), 10)
    : null;

  const offered = offeredSites(review.sites, recommendedRank);

  const run: RunSummary = {
    seed: review.seed,
    droneCount: review.droneCount,
    simDurationS: review.simDurationS ?? 0,
    mappedAtS: review.mappedAtS ?? null,
    coverageTotal: review.coverageTotal ?? 0,
    coverageGround: review.coverageGround ?? 0,
    verdict: review.verdict,
    justification: review.justification,
    startedAt: startedAt.toISOString(),
    addressMode: review.locationMode === "address" ? "address" : "random",
  };

  const report: DroneReport = {
    drones: mapDrones(review.tracks),
    events: mapEvents(review.events, startedAt),
    placementPhotos: mapPlacementPhotos(offered, recommendedRank),
    blockerPhotos: mapBlockerPhotos(offered),
    recommendation,
    video: review.videoKey ? { url: mediaUrl(review.videoKey) } : null,
    run,
  };

  return {
    id: review.id,
    name: review.name,
    address: review.address,
    email: review.email,
    reportStatus,
    createdAt: review.createdAt.toISOString(),
    report,
    sent: sentEmail,
  };
}
