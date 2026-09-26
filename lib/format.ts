import type { Blocker, BlockerSeverity, BlockerType } from "./types";

export function plural(count: number, one: string, many = `${one}s`): string {
  return count === 1 ? one : many;
}

export function firstName(name: string): string {
  return name.split(" ")[0] || name;
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** "Option A: garage sidewall" -> { option: "Option A", place: "Garage sidewall" } */
export function splitOptionLabel(label: string): {
  option: string;
  place: string | null;
} {
  const match = label.match(/^(Option [A-Z]+)\s*:\s*(.+)$/);
  if (!match) return { option: label, place: null };
  const place = match[2];
  return { option: match[1], place: place[0].toUpperCase() + place.slice(1) };
}

export const severityRank: Record<BlockerSeverity, number> = {
  High: 3,
  Medium: 2,
  Low: 1,
};

export function bySeverity(blockers: Blocker[]): Blocker[] {
  return [...blockers].sort(
    (a, b) => severityRank[b.severity] - severityRank[a.severity],
  );
}

/** The work each kind of blocker calls for, in the team's words. Shown as tags on a deal. */
export const blockerTag: Record<BlockerType, string> = {
  Vegetation: "Bush removal",
  "Communications Box": "Communications box removal",
  "Obstructed Meter": "Meter obstruction removal",
  "AC Unit": "AC unit clearance",
  "Gas Meter": "Gas meter clearance",
  "Unreadable Label": "New label photo",
  "Existing Solar/Generator": "Existing solar/generator review",
};

/** One tag per kind of work, most severe first. */
export function blockerTags(blockers: Blocker[]): string[] {
  return [...new Set(bySeverity(blockers).map((b) => blockerTag[b.type]))];
}

export function severityCounts(blockers: Blocker[]) {
  const counts = { total: blockers.length, High: 0, Medium: 0, Low: 0 };
  for (const blocker of blockers) counts[blocker.severity] += 1;
  const worst: BlockerSeverity | null = counts.High
    ? "High"
    : counts.Medium
      ? "Medium"
      : counts.Low
        ? "Low"
        : null;
  return { ...counts, worst };
}
