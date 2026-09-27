/** Unit tests for the deterministic parts of analysis.ts: mapping, severities, recommendation choice, and fallback text. No network calls (ANTHROPIC_API_KEY is never set here). */
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildBlockerDrafts,
  chooseRecommendedRank,
  classifyCandidate,
  severityForCandidate,
  severityForWarning,
  analyzeRun,
} from "./analysis.js";
import type { RunRecordSite } from "./schemas.js";
import type { RunPhoto } from "./schemas.js";

function makeSite(overrides: Partial<RunRecordSite> = {}): RunRecordSite {
  return {
    rank: 1,
    pos: [0, 0, 0],
    yaw: 0,
    meter: [1, 0, 0],
    cost: 1,
    breakdown: {},
    verdict: "pass",
    warnings: [],
    color: [0, 0, 0],
    ...overrides,
  };
}

function makePhoto(overrides: Partial<RunPhoto> = {}): RunPhoto {
  return {
    siteRank: 1,
    key: "reviews/x/photo-1.jpg",
    width: 800,
    height: 600,
    placementBoxes: [{ label: "Battery", box: { x: 40, y: 40, width: 20, height: 20 } }],
    candidates: [],
    ...overrides,
  };
}

test("classifyCandidate maps known classes and skips expected equipment", () => {
  assert.equal(classifyCandidate("BUSH"), "Vegetation");
  assert.equal(classifyCandidate("TREE"), "Vegetation");
  assert.equal(classifyCandidate("AC_UNIT"), "AC Unit");
  assert.equal(classifyCandidate("GAS_METER"), "Gas Meter");
  assert.equal(classifyCandidate("WINDOW"), "Window or Door");
  assert.equal(classifyCandidate("DOOR"), "Window or Door");
  assert.equal(classifyCandidate("GARAGE_DOOR"), "Window or Door");
  assert.equal(classifyCandidate("FENCE"), "Fence or Structure");
  assert.equal(classifyCandidate("SHED"), "Fence or Structure");
  assert.equal(classifyCandidate("PANEL"), null);
  assert.equal(classifyCandidate("CONDUIT"), null);
  assert.equal(classifyCandidate("METER"), null);
  assert.equal(classifyCandidate("SOMETHING_UNKNOWN"), null);
});

test("severityForCandidate: gas meter is High under 3 ft, Medium otherwise", () => {
  assert.equal(severityForCandidate("Gas Meter", 0.5), "High");
  assert.equal(severityForCandidate("Gas Meter", 0.913), "High");
  assert.equal(severityForCandidate("Gas Meter", 0.914), "Medium");
  assert.equal(severityForCandidate("Gas Meter", 2), "Medium");
});

test("severityForCandidate: window/door is High under 0.3m, Medium otherwise", () => {
  assert.equal(severityForCandidate("Window or Door", 0.1), "High");
  assert.equal(severityForCandidate("Window or Door", 0.3), "Medium");
  assert.equal(severityForCandidate("Window or Door", 1), "Medium");
});

test("severityForCandidate: vegetation is Low, Medium under 0.5m", () => {
  assert.equal(severityForCandidate("Vegetation", 0.2), "Medium");
  assert.equal(severityForCandidate("Vegetation", 0.5), "Low");
  assert.equal(severityForCandidate("Vegetation", 2), "Low");
});

test("severityForCandidate: AC unit always Low, fence/structure always Medium", () => {
  assert.equal(severityForCandidate("AC Unit", 0.01), "Low");
  assert.equal(severityForCandidate("AC Unit", 5), "Low");
  assert.equal(severityForCandidate("Fence or Structure", 0.01), "Medium");
  assert.equal(severityForCandidate("Fence or Structure", 5), "Medium");
});

test("severityForWarning: High on reject, Medium otherwise", () => {
  assert.equal(severityForWarning("reject"), "High");
  assert.equal(severityForWarning("manual_review"), "Medium");
  assert.equal(severityForWarning("pass"), "Medium");
});

test("buildBlockerDrafts turns candidates into blockers, skipping expected equipment", () => {
  const site = makeSite();
  const photo = makePhoto({
    candidates: [
      { trackId: 1, cls: "BUSH", box: { x: 10, y: 10, width: 5, height: 5 }, distanceM: 0.2 },
      { trackId: 2, cls: "PANEL", box: { x: 20, y: 20, width: 5, height: 5 }, distanceM: 0.1 },
    ],
  });
  const drafts = buildBlockerDrafts(site, photo);
  assert.equal(drafts.length, 1);
  assert.equal(drafts[0]?.type, "Vegetation");
  assert.equal(drafts[0]?.severity, "Medium");
  assert.equal(drafts[0]?.trackId, 1);
});

test("buildBlockerDrafts turns warnings into Clearance blockers anchored to the Battery box", () => {
  const site = makeSite({ verdict: "reject", warnings: ["Too close to the gas line easement"] });
  const photo = makePhoto();
  const drafts = buildBlockerDrafts(site, photo);
  assert.equal(drafts.length, 1);
  assert.equal(drafts[0]?.type, "Clearance");
  assert.equal(drafts[0]?.severity, "High");
  assert.equal(drafts[0]?.fallbackDescription, "Too close to the gas line easement");
  assert.deepEqual(drafts[0]?.box, { x: 40, y: 40, width: 20, height: 20 });
});

test("buildBlockerDrafts dedupes identical blockers", () => {
  const site = makeSite({ warnings: ["dup", "dup"] });
  const drafts = buildBlockerDrafts(site, makePhoto());
  assert.equal(drafts.length, 1);
});

test("buildBlockerDrafts falls back to a centered box with no placement overlays", () => {
  const site = makeSite({ warnings: ["no overlays"] });
  const photo = makePhoto({ placementBoxes: [] });
  const drafts = buildBlockerDrafts(site, photo);
  assert.deepEqual(drafts[0]?.box, { x: 35, y: 35, width: 30, height: 30 });
});

test("chooseRecommendedRank: lowest-rank pass wins", () => {
  const sites = [
    { rank: 1, verdict: "manual_review" as const },
    { rank: 2, verdict: "pass" as const },
    { rank: 3, verdict: "pass" as const },
  ];
  assert.equal(chooseRecommendedRank(sites), 2);
});

test("chooseRecommendedRank: falls back to lowest-rank manual_review when no pass", () => {
  const sites = [
    { rank: 1, verdict: "reject" as const },
    { rank: 2, verdict: "manual_review" as const },
    { rank: 3, verdict: "manual_review" as const },
  ];
  assert.equal(chooseRecommendedRank(sites), 2);
});

test("chooseRecommendedRank: falls back to rank 1 when nothing passes or qualifies", () => {
  const sites = [
    { rank: 1, verdict: "reject" as const },
    { rank: 2, verdict: "reject" as const },
  ];
  assert.equal(chooseRecommendedRank(sites), 1);
});

test("chooseRecommendedRank: rank 1 default with no sites at all", () => {
  assert.equal(chooseRecommendedRank([]), 1);
});

test("analyzeRun (no ANTHROPIC_API_KEY) uses deterministic fallback text throughout", async () => {
  const previousKey = process.env["ANTHROPIC_API_KEY"];
  delete process.env["ANTHROPIC_API_KEY"];
  try {
    const siteA = makeSite({
      rank: 1,
      verdict: "pass",
      warnings: [],
    });
    const siteB = makeSite({
      rank: 2,
      verdict: "manual_review",
      warnings: ["Encroaches on the setback line"],
    });
    const photoA = makePhoto({
      siteRank: 1,
      candidates: [{ trackId: 1, cls: "GAS_METER", box: { x: 1, y: 1, width: 2, height: 2 }, distanceM: 0.5 }],
    });
    const photoB = makePhoto({ siteRank: 2 });

    const result = await analyzeRun([siteA, siteB], new Map([[1, photoA], [2, photoB]]));

    assert.equal(result.recommendedRank, 1);
    assert.match(result.recommendationReason, /Site 1/);

    const analysisA = result.sites.find((s) => s.rank === 1);
    assert.ok(analysisA);
    assert.equal(analysisA.blockers.length, 1);
    assert.equal(analysisA.blockers[0]?.type, "Gas Meter");
    assert.equal(analysisA.blockers[0]?.severity, "High");
    assert.match(analysisA.summary, /Site 1/);

    const analysisB = result.sites.find((s) => s.rank === 2);
    assert.ok(analysisB);
    assert.equal(analysisB.blockers.length, 1);
    assert.equal(analysisB.blockers[0]?.type, "Clearance");
    assert.equal(analysisB.blockers[0]?.description, "Encroaches on the setback line");
  } finally {
    if (previousKey !== undefined) process.env["ANTHROPIC_API_KEY"] = previousKey;
  }
});
