/**
 * Response shapes the dashboard consumes, mirrored from
 * `src/dashboard/lib/types.ts` per ADR 0017's contract (that file is owned
 * by the dashboard app; this copy is what this service is required to
 * produce byte-for-byte, so a change on either side is a visible diff
 * rather than a silent drift).
 */

export type ReportStatus = "Awaiting Drone Report" | "Ready for Review" | "Email Sent";

export type BlockerType =
  | "Vegetation"
  | "AC Unit"
  | "Gas Meter"
  | "Communications Box"
  | "Obstructed Meter"
  | "Unreadable Label"
  | "Existing Solar/Generator"
  | "Window or Door"
  | "Fence or Structure"
  | "Clearance";

export type BlockerSeverity = "High" | "Medium" | "Low";

export type DroneStatus = "Flying" | "Charging" | "Idle" | "Failed";

export type FleetEventType = "capture" | "low_battery" | "failure" | "reassign" | "complete";

export type SiteVerdict = "pass" | "manual_review" | "reject";

/** Percentages of the image's width/height, so overlays stay aligned at any render size. */
export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Blocker {
  id: string;
  type: BlockerType;
  severity: BlockerSeverity;
  description: string;
  requiredFix: string;
  box: Box;
}

export interface PlacementOverlay {
  box: Box;
  label: string;
}

export interface PlacementPhoto {
  id: string;
  imageUrl: string;
  label: string;
  distanceToMeterFt: number;
  clearancePass: boolean;
  placementBoxes: PlacementOverlay[];
  verdict: SiteVerdict;
  recommended: boolean;
  summary: string;
  /** The candidate site this photo shows; matches `BlockerPhoto.siteRank`. */
  siteRank: number;
}

export interface BlockerPhoto {
  id: string;
  imageUrl: string;
  blockers: Blocker[];
  siteRank: number;
}

export interface Drone {
  id: string;
  status: DroneStatus;
  batteryPct: number;
  currentTask: string;
}

export interface FleetEvent {
  timestamp: string;
  droneId: string;
  type: FleetEventType;
  message: string;
  status?: DroneStatus;
  batteryPct?: number;
  currentTask?: string;
}

export interface RunSummary {
  seed: number;
  droneCount: number;
  simDurationS: number;
  mappedAtS: number | null;
  coverageTotal: number;
  coverageGround: number;
  verdict: string | null;
  justification: string | null;
  startedAt: string;
  addressMode: "random" | "address";
}

export interface DroneReport {
  drones: Drone[];
  events: FleetEvent[];
  placementPhotos: PlacementPhoto[];
  blockerPhotos: BlockerPhoto[];
  recommendation: { placementPhotoId: string; reason: string } | null;
  video: { url: string } | null;
  run: RunSummary;
}

export interface SentEmail {
  sentAt: string;
  subject: string;
  approved: boolean;
  placementPhotoId: string;
  blockerPhotoIds: string[];
}

export interface Member {
  id: string;
  name: string;
  address: string;
  email: string;
  reportStatus: ReportStatus;
  createdAt: string;
  report: DroneReport | null;
  sent: SentEmail | null;
}
