"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiError, listReviews, recordEmailSent } from "./api";
import { blockerTags, severityCounts } from "./format";
import type { Member, ReportStatus, SentEmail } from "./types";

/** A survey in progress shows "Awaiting Drone Report" until this catches it up. */
const POLL_MS = 5000;

interface MemberSessionContextValue {
  members: Member[];
  /** True only for the first load; a poll refresh never re-shows a loading state. */
  loading: boolean;
  error: string | null;
  getMember: (id: string) => Member | undefined;
  markEmailSent: (id: string, email: Omit<SentEmail, "sentAt">) => Promise<void>;
  refresh: () => Promise<void>;
}

const MemberSessionContext = createContext<MemberSessionContextValue | null>(
  null,
);

export function MemberSessionProvider({ children }: { children: ReactNode }) {
  const [members, setMembers] = useState<Member[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadedOnce = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const rows = await listReviews();
      setMembers(rows);
      setError(null);
    } catch (err) {
      // A failed poll keeps whatever the last successful list was; only the
      // first load has nothing to fall back to, which the error state covers.
      setError(
        err instanceof ApiError ? err.message : "Could not load reviews.",
      );
    } finally {
      loadedOnce.current = true;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // The mount-and-poll fetch pattern: `refresh`'s setState calls run in a
    // microtask after `await`, not synchronously inside this effect, so they
    // don't cause the render cascade the rule is guarding against. Its static
    // analysis can't tell sync from async apart and flags the call regardless.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const id = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const getMember = useCallback(
    (id: string) => members.find((m) => m.id === id),
    [members],
  );

  const markEmailSent = useCallback(
    async (id: string, email: Omit<SentEmail, "sentAt">) => {
      const updated = await recordEmailSent(id, email);
      setMembers((prev) => prev.map((m) => (m.id === id ? updated : m)));
    },
    [],
  );

  const value = useMemo(
    () => ({ members, loading, error, getMember, markEmailSent, refresh }),
    [members, loading, error, getMember, markEmailSent, refresh],
  );

  return (
    <MemberSessionContext.Provider value={value}>
      {children}
    </MemberSessionContext.Provider>
  );
}

export function useMemberSession() {
  const ctx = useContext(MemberSessionContext);
  if (!ctx) {
    throw new Error("useMemberSession must be used within MemberSessionProvider");
  }
  return ctx;
}

export type QueueStage = "ready" | "waiting" | "sent";

export const stageOf: Record<ReportStatus, QueueStage> = {
  "Ready for Review": "ready",
  "Awaiting Drone Report": "waiting",
  "Email Sent": "sent",
};

export function useQueue() {
  const { members, loading, error } = useMemberSession();
  const { rows, readyCount } = useMemo(() => {
    const rows = members.map((member) => {
      const report = member.report;
      // Count what's at the site the battery is going to (sent, else
      // recommended), matching what the member page shows for it.
      const placementId = member.sent?.placementPhotoId ?? report?.recommendation?.placementPhotoId;
      const siteRank = report?.placementPhotos.find((p) => p.id === placementId)?.siteRank;
      const blockers =
        report?.blockerPhotos
          .filter((p) => siteRank === undefined || p.siteRank === siteRank)
          .flatMap((p) => p.blockers) ?? [];
      return {
        id: member.id,
        name: member.name,
        address: member.address,
        email: member.email,
        stage: stageOf[member.reportStatus],
        blockers: severityCounts(blockers),
        tags: blockerTags(blockers),
        photos: report
          ? [...report.placementPhotos, ...report.blockerPhotos].map((p) => ({
              id: p.id,
              imageUrl: p.imageUrl,
            }))
          : [],
        sent: member.sent,
      };
    });

    const readyCount = rows.filter((r) => r.stage === "ready").length;

    return { rows, readyCount };
  }, [members]);

  return { rows, readyCount, loading, error };
}

export type QueueRow = ReturnType<typeof useQueue>["rows"][number];
