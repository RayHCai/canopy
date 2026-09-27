/**
 * Turns rule output (site verdicts, warnings, nearby detections) into
 * dashboard blockers and a recommended site (ADR 0017).
 *
 * The split that matters: everything a reviewer relies on to trust the
 * report -- which blockers exist, their severity, the recommended site --
 * is decided here in plain code from facts the site/perception stages
 * already produced. The LLM (Claude Haiku 4.5) is called only to phrase
 * `description`/`requiredFix`/`summary`/`reason` in plain language; it never
 * sees facts it wasn't given and never picks or invents a blocker or a
 * verdict. If it's unavailable or its output doesn't validate, the
 * deterministic text below is what ships -- a run must never get stuck in
 * "Awaiting Drone Report" for lack of an API key.
 */
import Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import type { Box, BlockerSeverity, BlockerType, SiteVerdict } from "./dashboardTypes.js";
import type { RunPhoto, RunRecordSite } from "./schemas.js";

const HAIKU_MODEL = "claude-haiku-4-5";
const LLM_TIMEOUT_MS = 15_000;

const GAS_METER_HIGH_M = 0.914; // 3 ft
const WINDOW_DOOR_HIGH_M = 0.3;
const VEGETATION_MEDIUM_M = 0.5;

/** A blocker before its prose is filled in -- the part the rules decide. */
export interface BlockerDraft {
  id: string;
  type: BlockerType;
  severity: BlockerSeverity;
  box: Box;
  trackId: number | null;
  /** Deterministic fallback text; may be replaced by the LLM's phrasing. */
  fallbackDescription: string;
  fallbackRequiredFix: string;
}

export interface ResolvedBlocker {
  id: string;
  type: BlockerType;
  severity: BlockerSeverity;
  description: string;
  requiredFix: string;
  box: Box;
  trackId: number | null;
}

export interface SiteAnalysis {
  rank: number;
  blockers: ResolvedBlocker[];
  summary: string;
}

export interface AnalysisResult {
  sites: SiteAnalysis[];
  recommendedRank: number;
  recommendationReason: string;
}

/** Detection classes that map to a dashboard blocker type; others (expected equipment) are skipped. */
const CLASS_TO_BLOCKER_TYPE: Record<string, BlockerType | undefined> = {
  BUSH: "Vegetation",
  TREE: "Vegetation",
  AC_UNIT: "AC Unit",
  GAS_METER: "Gas Meter",
  WINDOW: "Window or Door",
  DOOR: "Window or Door",
  GARAGE_DOOR: "Window or Door",
  FENCE: "Fence or Structure",
  SHED: "Fence or Structure",
  // Expected equipment near a meter/battery site -- not a blocker.
  PANEL: undefined,
  CONDUIT: undefined,
  METER: undefined,
};

/** Maps a candidate detection class to a blocker type, or null to skip it (expected equipment/unknown). */
export function classifyCandidate(cls: string): BlockerType | null {
  return CLASS_TO_BLOCKER_TYPE[cls] ?? null;
}

/** Severity for a detection-based blocker, purely from its type and distance to the site. */
export function severityForCandidate(type: BlockerType, distanceM: number): BlockerSeverity {
  switch (type) {
    case "Gas Meter":
      return distanceM < GAS_METER_HIGH_M ? "High" : "Medium";
    case "Window or Door":
      return distanceM < WINDOW_DOOR_HIGH_M ? "High" : "Medium";
    case "Vegetation":
      return distanceM < VEGETATION_MEDIUM_M ? "Medium" : "Low";
    case "AC Unit":
      return "Low";
    case "Fence or Structure":
      return "Medium";
    default:
      return "Medium";
  }
}

/** Severity for a site-warning ("Clearance") blocker: a rejected site's warnings are the reason it was rejected. */
export function severityForWarning(siteVerdict: SiteVerdict): BlockerSeverity {
  return siteVerdict === "reject" ? "High" : "Medium";
}

function fallbackTextForCandidate(type: BlockerType, distanceM: number): { description: string; requiredFix: string } {
  const distFt = (distanceM * 3.28084).toFixed(1);
  switch (type) {
    case "Vegetation":
      return {
        description: `Vegetation was detected about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Trim or clear the vegetation before installation so it does not encroach on the battery.",
      };
    case "AC Unit":
      return {
        description: `An AC unit sits about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Confirm the AC unit's service clearance is not blocked by the new battery.",
      };
    case "Gas Meter":
      return {
        description: `A gas meter is about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Maintain the required gas-meter clearance, or select a different site.",
      };
    case "Window or Door":
      return {
        description: `A window or door is about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Verify the battery will not obstruct egress, venting, or the opening's swing.",
      };
    case "Fence or Structure":
      return {
        description: `A fence or structure is about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Confirm there is enough access room between the structure and the battery for installation and service.",
      };
    default:
      return {
        description: `An obstruction was detected about ${distFt} ft from the proposed battery footprint.`,
        requiredFix: "Review this obstruction on site before installation.",
      };
  }
}

function findBatteryBox(placementBoxes: { label: string; box: Box }[]): Box {
  const battery = placementBoxes.find((p) => p.label.toLowerCase().includes("battery"));
  if (battery) return battery.box;
  const first = placementBoxes[0];
  if (first) return first.box;
  // No overlay to anchor to -- fall back to a centered placeholder box rather than crash.
  return { x: 35, y: 35, width: 30, height: 30 };
}

/** Builds the fact-only blocker drafts for one site: candidate detections plus warning strings. Deduped. */
export function buildBlockerDrafts(site: RunRecordSite, photo: RunPhoto | undefined): BlockerDraft[] {
  const drafts: BlockerDraft[] = [];
  const seen = new Set<string>();
  let n = 0;

  if (photo) {
    for (const candidate of photo.candidates) {
      const type = classifyCandidate(candidate.cls);
      if (!type) continue;
      const severity = severityForCandidate(type, candidate.distanceM);
      const { description, requiredFix } = fallbackTextForCandidate(type, candidate.distanceM);
      const dedupeKey = JSON.stringify({ type, box: candidate.box, description });
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);
      drafts.push({
        id: `s${site.rank}-b${n++}`,
        type,
        severity,
        box: candidate.box,
        trackId: candidate.trackId,
        fallbackDescription: description,
        fallbackRequiredFix: requiredFix,
      });
    }
  }

  for (const warning of site.warnings) {
    const severity = severityForWarning(site.verdict);
    const box = findBatteryBox(photo?.placementBoxes ?? []);
    const dedupeKey = JSON.stringify({ type: "Clearance", box, description: warning });
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    drafts.push({
      id: `s${site.rank}-b${n++}`,
      type: "Clearance",
      severity,
      box,
      trackId: null,
      fallbackDescription: warning,
      fallbackRequiredFix: "Resolve this clearance issue before approving the site.",
    });
  }

  return drafts;
}

/**
 * Deterministic best site: lowest rank with a "pass" verdict, else lowest
 * rank "manual_review", else rank 1. Never delegated to the LLM (ADR 0017).
 */
export function chooseRecommendedRank(sites: { rank: number; verdict: SiteVerdict }[]): number {
  const byRank = [...sites].sort((a, b) => a.rank - b.rank);
  const pass = byRank.find((s) => s.verdict === "pass");
  if (pass) return pass.rank;
  const manualReview = byRank.find((s) => s.verdict === "manual_review");
  if (manualReview) return manualReview.rank;
  return byRank[0]?.rank ?? 1;
}

function fallbackSiteSummary(site: RunRecordSite, blockerCount: number): string {
  const verdictText =
    site.verdict === "pass"
      ? "passes every clearance rule"
      : site.verdict === "manual_review"
        ? "needs a reviewer's sign-off"
        : "is not recommended";
  const blockerText =
    blockerCount === 0
      ? "no blockers found nearby"
      : `${blockerCount} potential blocker${blockerCount === 1 ? "" : "s"} nearby`;
  return `Site ${site.rank} ${verdictText}, with ${blockerText}.`;
}

function fallbackRecommendationReason(site: RunRecordSite): string {
  if (site.verdict === "pass") {
    return `Site ${site.rank} is recommended: it is the highest-ranked site that passed every clearance rule.`;
  }
  if (site.verdict === "manual_review") {
    return `Site ${site.rank} is recommended for a manual look -- no site passed every rule outright, and this is the best candidate.`;
  }
  return `Site ${site.rank} is offered as a fallback; no site passed or qualified for manual review.`;
}

// --- LLM prose (optional) -----------------------------------------------

const llmBlockerTextSchema = z.object({ description: z.string().min(1), requiredFix: z.string().min(1) });
const llmResponseSchema = z.object({
  blockers: z.record(z.string(), llmBlockerTextSchema).optional(),
  siteSummaries: z.record(z.string(), z.string().min(1)).optional(),
  recommendationReason: z.string().min(1).optional(),
});

interface LlmFacts {
  sites: {
    rank: number;
    verdict: SiteVerdict;
    cost: number;
    breakdown: Record<string, number | null>;
    warnings: string[];
    blockers: { id: string; type: BlockerType; severity: BlockerSeverity; distanceM: number | null }[];
  }[];
  recommendedRank: number;
  overallVerdict: string | null;
  overallJustification: string | null;
}

async function callHaikuForProse(
  facts: LlmFacts,
): Promise<{ blockers: Record<string, { description: string; requiredFix: string }>; siteSummaries: Record<string, string>; recommendationReason: string | null } | null> {
  const apiKey = process.env["ANTHROPIC_API_KEY"];
  if (!apiKey) return null;

  const client = new Anthropic({ apiKey });
  const prompt = [
    "You write short, plain-language homeowner/installer-facing text for a solar battery site survey report.",
    "You are given structured facts only -- rule verdicts, clearance-rule numbers, warnings, and a list of blockers with distances.",
    "Never invent a blocker, distance, or verdict that isn't in the facts. Never change a verdict or the recommendation.",
    "",
    "Return ONLY a JSON object of this shape (no prose outside the JSON):",
    '{"blockers": {"<blockerId>": {"description": "one sentence", "requiredFix": "one sentence a homeowner or installer can act on"}}, ' +
      '"siteSummaries": {"<rank>": "1-2 sentence summary of this site"}, ' +
      '"recommendationReason": "1-2 sentences on why the recommended site was chosen"}',
    "",
    "Facts:",
    JSON.stringify(facts, null, 2),
  ].join("\n");

  try {
    const response = await client.messages.create(
      {
        model: HAIKU_MODEL,
        max_tokens: 2048,
        messages: [{ role: "user", content: prompt }],
      },
      { timeout: LLM_TIMEOUT_MS },
    );
    const textBlock = response.content.find((b): b is Anthropic.TextBlock => b.type === "text");
    if (!textBlock) return null;
    // Claude may wrap JSON in a fenced code block despite the instruction; strip it defensively.
    const raw = textBlock.text.trim().replace(/^```(?:json)?\s*/, "").replace(/```\s*$/, "");
    const parsedJson: unknown = JSON.parse(raw);
    const validated = llmResponseSchema.safeParse(parsedJson);
    if (!validated.success) return null;
    return {
      blockers: validated.data.blockers ?? {},
      siteSummaries: validated.data.siteSummaries ?? {},
      recommendationReason: validated.data.recommendationReason ?? null,
    };
  } catch {
    // Network error, timeout, non-JSON response, refusal, etc. -- the deterministic
    // fallback text carries the report; a broken LLM call must never block a run.
    return null;
  }
}

/**
 * Runs the full per-site analysis: deterministic blockers and recommendation
 * first, then an optional LLM pass to phrase the prose. Falls back to
 * deterministic text for any item the LLM omits or that fails validation.
 */
export async function analyzeRun(
  sites: RunRecordSite[],
  photosByRank: Map<number, RunPhoto>,
): Promise<AnalysisResult> {
  const draftsByRank = new Map<number, BlockerDraft[]>();
  for (const site of sites) {
    draftsByRank.set(site.rank, buildBlockerDrafts(site, photosByRank.get(site.rank)));
  }

  const recommendedRank = chooseRecommendedRank(sites.map((s) => ({ rank: s.rank, verdict: s.verdict })));
  const recommendedSite = sites.find((s) => s.rank === recommendedRank);

  const facts: LlmFacts = {
    sites: sites.map((site) => ({
      rank: site.rank,
      verdict: site.verdict,
      cost: site.cost,
      breakdown: site.breakdown,
      warnings: site.warnings,
      blockers: (draftsByRank.get(site.rank) ?? []).map((d) => ({
        id: d.id,
        type: d.type,
        severity: d.severity,
        distanceM: photosByRank.get(site.rank)?.candidates.find((c) => c.trackId === d.trackId)?.distanceM ?? null,
      })),
    })),
    recommendedRank,
    overallVerdict: null,
    overallJustification: null,
  };

  const llm = await callHaikuForProse(facts);

  const resultSites: SiteAnalysis[] = sites.map((site) => {
    const drafts = draftsByRank.get(site.rank) ?? [];
    const blockers: ResolvedBlocker[] = drafts.map((d) => {
      const llmText = llm?.blockers[d.id];
      return {
        id: d.id,
        type: d.type,
        severity: d.severity,
        description: llmText?.description ?? d.fallbackDescription,
        requiredFix: llmText?.requiredFix ?? d.fallbackRequiredFix,
        box: d.box,
        trackId: d.trackId,
      };
    });
    const summary = llm?.siteSummaries[String(site.rank)] ?? fallbackSiteSummary(site, blockers.length);
    return { rank: site.rank, blockers, summary };
  });

  const recommendationReason =
    llm?.recommendationReason ?? (recommendedSite ? fallbackRecommendationReason(recommendedSite) : "");

  return { sites: resultSites, recommendedRank, recommendationReason };
}
