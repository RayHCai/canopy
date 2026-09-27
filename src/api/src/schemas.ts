/**
 * zod schemas for every request body this service accepts.
 *
 * The `record` schema mirrors `ViewerSession.run_record()` (Python side,
 * ADR 0017) field for field, but every object uses `.passthrough()` so an
 * extra key Python adds later doesn't 400 the upload -- we store `record`
 * faithfully (see mapping.ts) and only rely on the fields we actually read.
 */
import { z } from "zod";

export const createReviewSchema = z.object({
  name: z.string().min(1),
  email: z.string().min(1),
  address: z.string().min(1),
  lat: z.number().optional(),
  lon: z.number().optional(),
  siteId: z.string().optional(),
  droneCount: z.number().int().positive(),
  seed: z.number().int(),
});
export type CreateReviewInput = z.infer<typeof createReviewSchema>;

export const uploadsRequestSchema = z.object({
  files: z
    .array(
      z.object({
        name: z.string().min(1),
        contentType: z.string().min(1),
      }),
    )
    .min(1),
});
export type UploadsRequestInput = z.infer<typeof uploadsRequestSchema>;

const vec3Schema = z.tuple([z.number(), z.number(), z.number()]);
const boxSchema = z
  .object({
    x: z.number(),
    y: z.number(),
    width: z.number(),
    height: z.number(),
  })
  .passthrough();

const trackSchema = z
  .object({
    id: z.number().int(),
    samples: z.array(z.tuple([z.number(), z.number(), z.number(), z.number(), z.number(), z.number()])),
    alive: z.boolean(),
    battery: z.number(),
  })
  .passthrough();

const fleetEventSchema = z
  .object({
    t: z.number(),
    drone_id: z.number().int(),
    type: z.enum(["capture", "low_battery", "failure", "complete"]),
    message: z.string(),
    status: z.enum(["Flying", "Charging", "Idle", "Failed"]),
    battery_pct: z.number().int(),
    task: z.string(),
  })
  .passthrough();

const detectionSchema = z
  .object({
    id: z.number().int(),
    cls: z.string(),
    color: vec3Schema,
    center: vec3Schema,
    size: vec3Schema,
    yaw: z.number(),
    confidence: z.number(),
  })
  .passthrough();

const siteSuggestionSchema = z
  .object({
    rank: z.number().int(),
    pos: vec3Schema,
    yaw: z.number(),
    meter: vec3Schema,
    cost: z.number(),
    breakdown: z.record(z.string(), z.number().nullable()),
    verdict: z.enum(["pass", "manual_review", "reject"]),
    warnings: z.array(z.string()),
    color: vec3Schema,
  })
  .passthrough();

const siteAssessmentSchema = z
  .object({
    sites: z.array(siteSuggestionSchema),
    message: z.string().nullable(),
    battery: vec3Schema.nullable(),
    verdict: z.string().optional(),
    justification: z.string().optional(),
  })
  .passthrough();

export const runRecordSchema = z
  .object({
    seed: z.number().int(),
    drones: z.number().int(),
    sim_time_s: z.number(),
    phase: z.string(),
    mapped_at_s: z.number().nullable(),
    coverage: z
      .object({
        ground_band: z.number(),
        total: z.number(),
      })
      .passthrough(),
    lot: z.tuple([vec3Schema, vec3Schema]),
    location: z
      .object({
        mode: z.enum(["random", "address"]),
        label: z.string().nullable(),
        site_id: z.string().nullable(),
      })
      .passthrough(),
    tracks: z.array(trackSchema),
    events: z.array(fleetEventSchema),
    detections: z.array(detectionSchema),
    site: siteAssessmentSchema.nullable(),
  })
  .passthrough();
export type RunRecord = z.infer<typeof runRecordSchema>;
export type RunRecordSite = z.infer<typeof siteSuggestionSchema>;
export type RunRecordDetection = z.infer<typeof detectionSchema>;

export const photoCandidateSchema = z
  .object({
    trackId: z.number().int(),
    cls: z.string(),
    box: boxSchema,
    distanceM: z.number(),
  })
  .passthrough();

export const photoSchema = z
  .object({
    siteRank: z.number().int(),
    key: z.string().min(1),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    placementBoxes: z.array(
      z
        .object({
          label: z.string(),
          box: boxSchema,
        })
        .passthrough(),
    ),
    candidates: z.array(photoCandidateSchema),
  })
  .passthrough();
export type RunPhoto = z.infer<typeof photoSchema>;

export const runRequestSchema = z.object({
  record: runRecordSchema,
  photos: z.array(photoSchema),
  videoKey: z.string().nullable(),
});
export type RunRequestInput = z.infer<typeof runRequestSchema>;

export const emailRequestSchema = z.object({
  subject: z.string().min(1),
  approved: z.boolean(),
  placementPhotoId: z.string().min(1),
  blockerPhotoIds: z.array(z.string()),
});
export type EmailRequestInput = z.infer<typeof emailRequestSchema>;
