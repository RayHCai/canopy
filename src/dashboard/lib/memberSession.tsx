"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { blockerTags, severityCounts } from "./format";
import { mockMembers } from "./mockData";
import type { Member, ReportStatus, SentEmail } from "./types";

interface MemberSessionContextValue {
  members: Member[];
  sent: Record<string, SentEmail>;
  getMember: (id: string) => Member | undefined;
  markEmailSent: (id: string, email: Omit<SentEmail, "sentAt">) => void;
}

const MemberSessionContext = createContext<MemberSessionContextValue | null>(
  null,
);

export function MemberSessionProvider({ children }: { children: ReactNode }) {
  const [members, setMembers] = useState<Member[]>(() =>
    structuredClone(mockMembers),
  );
  const [sent, setSent] = useState<Record<string, SentEmail>>({});

  const getMember = useCallback(
    (id: string) => members.find((m) => m.id === id),
    [members],
  );

  const markEmailSent = useCallback(
    (id: string, email: Omit<SentEmail, "sentAt">) => {
      setMembers((prev) =>
        prev.map((m) => (m.id === id ? { ...m, reportStatus: "Email Sent" } : m)),
      );
      setSent((prev) => ({
        ...prev,
        [id]: { ...email, sentAt: new Date().toISOString() },
      }));
    },
    [],
  );

  const value = useMemo(
    () => ({ members, sent, getMember, markEmailSent }),
    [members, sent, getMember, markEmailSent],
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
  const { members, sent } = useMemberSession();
  return useMemo(() => {
    const rows = members.map((member) => {
      const report = member.report;
      const blockers = report?.blockerPhotos.flatMap((p) => p.blockers) ?? [];
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
        sent: sent[member.id] ?? null,
      };
    });

    const readyCount = rows.filter((r) => r.stage === "ready").length;

    return { rows, readyCount };
  }, [members, sent]);
}

export type QueueRow = ReturnType<typeof useQueue>["rows"][number];
