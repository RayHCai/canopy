"use client";

import { BlockerReview } from "@/components/member/BlockerReview";
import { LetterPane, type SendPhase } from "@/components/member/LetterPane";
import { MemberHeader } from "@/components/member/MemberHeader";
import { PlacementChooser } from "@/components/member/PlacementChooser";
import { Step } from "@/components/member/Step";
import { delay, prefersReducedMotion } from "@/components/ui/motion";
import { buildMemberEmail, includesBlockers } from "@/lib/buildMemberEmail";
import { firstName as firstNameOf, plural, splitOptionLabel } from "@/lib/format";
import { useMemberSession } from "@/lib/memberSession";
import { sendEmail } from "@/lib/sendEmail";
import type { BlockerPhoto, DroneReport, Member, PlacementPhoto } from "@/lib/types";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

/**
 * A copy of the photo arcs from where you clicked into the letter's photo row,
 * so the connection between choosing and the email changing is visible.
 */
function flyToLetter(fromEl: HTMLElement | null, attachKey: string, src: string) {
  if (!fromEl || prefersReducedMotion()) return;
  requestAnimationFrame(() => {
    const target = document.querySelector<HTMLElement>(`[data-attach="${attachKey}"]`);
    if (!target) return;
    const a = fromEl.getBoundingClientRect();
    const b = target.getBoundingClientRect();
    if (b.width === 0 || b.bottom < 0 || b.top > window.innerHeight) return;

    const ghost = document.createElement("img");
    ghost.src = src;
    ghost.alt = "";
    Object.assign(ghost.style, {
      position: "fixed",
      left: `${a.left}px`,
      top: `${a.top}px`,
      width: `${a.width}px`,
      height: `${a.height}px`,
      objectFit: "cover",
      borderRadius: "3px",
      zIndex: "70",
      pointerEvents: "none",
      transformOrigin: "0 0",
      boxShadow: "0 24px 48px -16px rgba(22, 21, 15, 0.45)",
    });
    document.body.appendChild(ghost);

    const dx = b.left - a.left;
    const dy = b.top - a.top;
    const sx = b.width / a.width;
    const sy = b.height / a.height;
    const flight = ghost.animate(
      [
        { transform: "none", opacity: 0.95 },
        {
          transform: `translate(${dx * 0.55}px, ${dy * 0.55 - 90}px) scale(${(1 + sx) / 2.4}, ${(1 + sy) / 2.4})`,
          opacity: 0.95,
          offset: 0.5,
        },
        { transform: `translate(${dx}px, ${dy}px) scale(${sx}, ${sy})`, opacity: 0.6 },
      ],
      { duration: 760, easing: "cubic-bezier(0.45, 0, 0.2, 1)" },
    );
    flight.finished.finally(() => ghost.remove()).catch(() => {});
    target.animate(
      [
        { opacity: 0, transform: "scale(0.6)" },
        { opacity: 0, transform: "scale(0.6)", offset: 0.8 },
        { opacity: 1, transform: "scale(1.15)", offset: 0.92 },
        { opacity: 1, transform: "scale(1)" },
      ],
      { duration: 900, easing: "ease-out" },
    );
  });
}

function MobileSendBar({ text, sent }: { text: string; sent: boolean }) {
  const [letterVisible, setLetterVisible] = useState(false);

  useEffect(() => {
    const letter = document.getElementById("letter");
    if (!letter) return;
    const observer = new IntersectionObserver(([entry]) => setLetterVisible(entry.isIntersecting), {
      rootMargin: "0px 0px -30% 0px",
    });
    observer.observe(letter);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className={`fixed inset-x-0 bottom-0 z-30 border-t border-rule bg-paper/90 pb-[env(safe-area-inset-bottom)] backdrop-blur-md transition-transform duration-500 ease-snap xl:hidden ${
        letterVisible ? "translate-y-full" : ""
      }`}
    >
      <div className="mx-auto flex max-w-[1320px] items-center justify-between gap-4 px-5 py-3 sm:px-10">
        <p className="min-w-0 truncate text-[14px] text-ink-2">{text}</p>
        <a
          href="#letter"
          className="inline-flex h-11 shrink-0 items-center gap-2 rounded-[3px] bg-ink px-4 text-[15px] font-semibold text-white"
        >
          {sent ? "See email" : "Review email"} <span aria-hidden>↓</span>
        </a>
      </div>
    </div>
  );
}

function Review({ member, report }: { member: Member; report: DroneReport }) {
  const { members, markEmailSent } = useMemberSession();
  const sentEmail = member.sent;
  const [placementId, setPlacementId] = useState<string | null>(
    sentEmail?.placementPhotoId ?? report.recommendation?.placementPhotoId ?? null,
  );
  const [includedIds, setIncludedIds] = useState<Set<string>>(
    () => new Set(sentEmail?.blockerPhotoIds ?? []),
  );
  const [phase, setPhase] = useState<SendPhase>("idle");
  const [stampNow, setStampNow] = useState(false);

  const locked = sentEmail !== null || phase === "sending";
  const first = firstNameOf(member.name);

  const placement = report.placementPhotos.find((p) => p.id === placementId) ?? null;
  // Only what stands in the way at the chosen site matters to this member, so
  // blockers follow the placement; a pick at another site drops out with it.
  const siteReport = useMemo(
    () => ({
      ...report,
      blockerPhotos: placement
        ? report.blockerPhotos.filter((p) => p.siteRank === placement.siteRank)
        : [],
    }),
    [report, placement],
  );
  const included = useMemo(
    () => siteReport.blockerPhotos.filter((p) => includedIds.has(p.id)),
    [siteReport.blockerPhotos, includedIds],
  );
  const draft = useMemo(
    () => (placement ? buildMemberEmail(member, placement, included) : null),
    [member, placement, included],
  );
  const next =
    members.find((m) => m.id !== member.id && m.reportStatus === "Ready for Review") ?? null;

  const choosePlacement = (photo: PlacementPhoto, photoEl: HTMLElement | null) => {
    if (locked || photo.id === placementId) return;
    setPlacementId(photo.id);
    setPhase("idle");
    flyToLetter(photoEl, "placement", photo.imageUrl);
  };

  const toggleBlockerPhoto = (photo: BlockerPhoto, photoEl: HTMLElement | null) => {
    if (locked) return;
    const adding = !includedIds.has(photo.id);
    setIncludedIds((prev) => {
      const nextIds = new Set(prev);
      if (adding) nextIds.add(photo.id);
      else nextIds.delete(photo.id);
      return nextIds;
    });
    setPhase("idle");
    if (adding) flyToLetter(photoEl, photo.id, photo.imageUrl);
  };

  const confirmSend = async () => {
    if (!draft || !placement) return;
    setPhase("sending");
    try {
      await sendEmail(draft);
      await markEmailSent(member.id, {
        subject: draft.subject,
        approved: !includesBlockers(included),
        placementPhotoId: placement.id,
        blockerPhotoIds: included.map((p) => p.id),
      });
    } catch {
      setPhase("error");
      return;
    }
    setStampNow(true);
    setPhase("idle");
  };

  const barText = sentEmail
    ? `Sent to ${first}`
    : placement
      ? `${splitOptionLabel(placement.label).option} · ${included.length} blocker ${plural(
          included.length,
          "photo",
        )}`
      : "Choose a placement to start the email";

  return (
    <>
      <div className="mt-14 grid gap-x-14 gap-y-24 xl:grid-cols-[minmax(0,1fr)_392px]">
        <div className="min-w-0 space-y-28">
          <div className="anim-rise" style={delay(80)}>
            <Step
              n="01"
              done={placement !== null}
              title="Battery Placement"
              meta={
                sentEmail
                  ? "Sent · locked"
                  : report.placementPhotos.length > 1
                    ? "Required · choose one"
                    : "Required · confirm it"
              }
            >
              <PlacementChooser
                name={`placement-${member.id}`}
                photos={report.placementPhotos}
                selectedId={placementId}
                locked={locked}
                recommendation={report.recommendation}
                onSelect={choosePlacement}
              />
            </Step>
          </div>

          <div className="anim-rise" style={delay(140)}>
            <Step
              n="02"
              title="Blockers"
              meta={
                sentEmail
                  ? "Sent · locked"
                  : !placement
                    ? "Choose a placement first"
                    : siteReport.blockerPhotos.length > 0
                      ? "Optional · include any"
                      : "Nothing flagged"
              }
            >
              {placement ? (
                <BlockerReview
                  key={placement.siteRank}
                  firstName={first}
                  photos={siteReport.blockerPhotos}
                  includedIds={includedIds}
                  locked={locked}
                  onToggle={toggleBlockerPhoto}
                />
              ) : (
                <p className="border-t border-rule py-6 text-[17px] text-ink-3">
                  Blockers show up here for the site you choose in step 01.
                </p>
              )}
            </Step>
          </div>
        </div>

        <LetterPane
          member={member}
          firstName={first}
          report={siteReport}
          placement={placement}
          included={included}
          includedIds={includedIds}
          draft={draft}
          sentEmail={sentEmail}
          stampNow={stampNow}
          phase={phase}
          next={next}
          onRequestSend={() => setPhase("confirm")}
          onConfirmSend={confirmSend}
          onCancelSend={() => setPhase("idle")}
        />
      </div>

      <MobileSendBar text={barText} sent={sentEmail !== null} />
    </>
  );
}

function NoReport() {
  return (
    <section className="anim-rise mt-14 border-t-2 border-ink pt-8">
      <p className="type-display max-w-[16ch] text-[clamp(34px,4.6vw,64px)] text-ink-3">
        No survey report yet.
      </p>
      <p className="mt-6 max-w-[54ch] text-[18px] leading-relaxed text-ink-2">
        The drones haven’t delivered a report for this home, so there is nothing to review or send.
        It will show up here when it arrives.
      </p>
      <Link
        href="/"
        transitionTypes={["nav-back"]}
        className="group mt-10 inline-flex items-center gap-3 text-[17px] font-semibold"
      >
        <span aria-hidden className="transition-transform duration-500 ease-snap group-hover:-translate-x-1.5">
          ←
        </span>
        Back to the queue
      </Link>
    </section>
  );
}

export function MemberClient({ initialMember }: { initialMember: Member }) {
  const { getMember } = useMemberSession();
  // The session's own poll may not have this member yet (it was just opened
  // from a fresh server fetch); fall back to what the page already fetched.
  const member = getMember(initialMember.id) ?? initialMember;

  return (
    <>
      <MemberHeader member={member} sentEmail={member.sent} />
      {member.report ? (
        <Review key={member.id} member={member} report={member.report} />
      ) : (
        <NoReport />
      )}
    </>
  );
}
