export type ReportStatus =
  | "Awaiting Drone Report"
  | "Ready for Review"
  | "Email Sent";

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

export type FleetEventType =
  | "capture"
  | "low_battery"
  | "failure"
  | "reassign"
  | "complete";

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

/** The site stage's own outcome for a candidate site (ADR: per-rule measures, three-way verdict). */
export type PlacementVerdict = "pass" | "manual_review" | "reject";

export interface PlacementPhoto {
  id: string;
  imageUrl: string;
  label: string;
  distanceToMeterFt: number;
  clearancePass: boolean;
  placementBoxes: PlacementOverlay[];
  verdict: PlacementVerdict;
  /** Whether this is the site the API's analysis recommended. */
  recommended: boolean;
  /** LLM or deterministic prose over the rule facts for this site. */
  summary: string;
  /** The candidate site this photo shows; matches `BlockerPhoto.siteRank`. */
  siteRank: number;
}

export interface BlockerPhoto {
  id: string;
  imageUrl: string;
  blockers: Blocker[];
  /** Which candidate site (1-indexed, matches PlacementPhoto option order) this photo is at. */
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
  /**
   * Replay-only state deltas. Kept alongside the human-readable `message`
   * instead of parsing it back apart at playback time.
   */
  status?: DroneStatus;
  batteryPct?: number;
  currentTask?: string;
  targetDroneId?: string;
  targetStatus?: DroneStatus;
  targetTask?: string;
}

/** Facts about the run itself, independent of any one candidate site. */
export interface RunSummary {
  seed: number;
  droneCount: number;
  simDurationS: number;
  /** Simulated seconds until the swarm considered the house mapped; null if it never did. */
  mappedAtS: number | null;
  coverageTotal: number;
  coverageGround: number;
  verdict: string | null;
  justification: string | null;
  startedAt: string;
  /** "address" surveyed the member's real house; "random" flew a generated stand-in. */
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

export interface MemberEmailDraft {
  subject: string;
  body: string;
  html: string;
  to: string;
  placementPhoto: PlacementPhoto;
  blockerPhotos: BlockerPhoto[];
}

/** What actually went out, kept so the member page can show it after the fact. */
export interface SentEmail {
  sentAt: string;
  subject: string;
  approved: boolean;
  placementPhotoId: string;
  blockerPhotoIds: string[];
}
