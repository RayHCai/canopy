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
  | "Existing Solar/Generator";

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

export interface PlacementPhoto {
  id: string;
  imageUrl: string;
  label: string;
  distanceToMeterFt: number;
  clearancePass: boolean;
  placementBoxes: PlacementOverlay[];
}

export interface BlockerPhoto {
  id: string;
  imageUrl: string;
  blockers: Blocker[];
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

export interface DroneReport {
  drones: Drone[];
  events: FleetEvent[];
  placementPhotos: PlacementPhoto[];
  blockerPhotos: BlockerPhoto[];
}

export interface Member {
  id: string;
  name: string;
  address: string;
  email: string;
  reportStatus: ReportStatus;
  report: DroneReport | null;
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
