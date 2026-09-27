import { getMemberById } from "./mockData";
import type { DroneReport } from "./types";

const MOCK_LOAD_DELAY_MS = 500;

/**
 * TODO: swap this body for a fetch against `/sim-output/${memberId}.json`
 * once the drone simulation writes real reports into /public/sim-output.
 * The return shape already matches DroneReport, so no caller needs to change.
 */
export async function loadReport(memberId: string): Promise<DroneReport | null> {
  await new Promise((resolve) => setTimeout(resolve, MOCK_LOAD_DELAY_MS));
  return getMemberById(memberId)?.report ?? null;
}
