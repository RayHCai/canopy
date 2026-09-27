"use client";

import { PrimaryButton, QuietButton } from "@/components/ui/Button";
import { CheckGlyph, CrossGlyph, WarnGlyph } from "@/components/ui/glyphs";
import { delay, prefersReducedMotion } from "@/components/ui/motion";
import {
  SUBJECT_ACTION_NEEDED,
  SUBJECT_APPROVED,
  collectFixes,
  includesBlockers,
} from "@/lib/buildMemberEmail";
import { formatTime, plural, splitOptionLabel } from "@/lib/format";
import type {
  BlockerPhoto,
  DroneReport,
  Member,
  MemberEmailDraft,
  PlacementPhoto,
  SentEmail,
} from "@/lib/types";
import Image from "next/image";
import Link from "next/link";
import { useId, useLayoutEffect, useRef } from "react";

export type SendPhase = "idle" | "confirm" | "sending" | "error";

/**
 * The email's real HTML, rendered in a shadow root so the app's styles can't
 * touch it: what you read here is exactly what the member receives.
 */
function EmailBody({ html }: { html: string }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const rendered = useRef(false);

  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const root = host.shadowRoot ?? host.attachShadow({ mode: "open" });
    root.innerHTML = `<style>:host{all:initial;display:block}</style>${html}`;
    if (rendered.current && !prefersReducedMotion()) {
      host.animate(
        [
          { opacity: 0.3, filter: "blur(3px)" },
          { opacity: 1, filter: "blur(0)" },
        ],
        { duration: 460, easing: "cubic-bezier(0.22, 1, 0.36, 1)" },
      );
    }
    rendered.current = true;
  }, [html]);

  return <div ref={hostRef} style={{ zoom: 0.84 }} />;
}

function Postmark({ sentAt, animate }: { sentAt: string; animate: boolean }) {
  const ringId = `postmark-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const sent = new Date(sentAt);
  const time = sent.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const date = sent.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });

  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute right-2 top-[92px] h-[150px] w-[150px] text-signal mix-blend-multiply ${
        animate ? "animate-[stamp_760ms_var(--ease-snap)_both]" : "rotate-[-8deg] opacity-90"
      }`}
    >
      <svg viewBox="0 0 160 160" className="h-full w-full">
        <defs>
          <path id={ringId} d="M80 80 m-61 0 a61 61 0 1 1 122 0 a61 61 0 1 1 -122 0" />
        </defs>
        <circle cx="80" cy="80" r="76" fill="none" stroke="currentColor" strokeWidth="3.5" />
        <circle cx="80" cy="80" r="47" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <text fill="currentColor" fontSize="10.5" letterSpacing="2.4" className="font-mono">
          <textPath href={`#${ringId}`}>CANOPY · SITE SURVEY REVIEW · BASE POWER ·</textPath>
        </text>
        <text
          x="80"
          y="80"
          textAnchor="middle"
          fill="currentColor"
          fontSize="25"
          fontWeight="800"
          className="font-sans [font-stretch:125%]"
        >
          SENT
        </text>
        <text x="80" y="98" textAnchor="middle" fill="currentColor" fontSize="11" className="font-mono">
          {time.toUpperCase()}
        </text>
        <text x="80" y="112" textAnchor="middle" fill="currentColor" fontSize="9.5" className="font-mono">
          {date.toUpperCase()}
        </text>
      </svg>
    </div>
  );
}

function Thumb({ attach, src, chosen = false }: { attach: string; src: string; chosen?: boolean }) {
  return (
    <span
      data-attach={attach}
      className={`block h-9 w-12 overflow-hidden rounded-[2px] ${
        chosen ? "ring-2 ring-signal ring-offset-2 ring-offset-paper" : "ring-1 ring-ink/15"
      }`}
    >
      <Image src={src} alt="" width={48} height={36} unoptimized className="h-full w-full object-cover" />
    </span>
  );
}

function nudgesFor({
  firstName,
  report,
  placement,
  includedIds,
  sent,
}: {
  firstName: string;
  report: DroneReport;
  placement: PlacementPhoto | null;
  includedIds: Set<string>;
  sent: boolean;
}) {
  const nudges: { id: string; tone: "critical" | "warning"; text: string }[] = [];
  if (!placement) return nudges;

  if (placement.verdict === "reject") {
    const { option } = splitOptionLabel(placement.label);
    nudges.push({
      id: "clearance",
      tone: "critical",
      text: sent
        ? `${option} failed the clearance check, and the email recommended it.`
        : `${option} failed the clearance check. The email will still recommend it.`,
    });
  } else if (placement.verdict === "manual_review") {
    const { option } = splitOptionLabel(placement.label);
    nudges.push({
      id: "clearance",
      tone: "warning",
      text: sent
        ? `${option} needed a reviewer's sign-off on clearance, and the email recommended it.`
        : `${option} needs your sign-off on clearance before the email recommends it.`,
    });
  }

  const allFlagged = report.blockerPhotos.flatMap((p) => p.blockers);
  const leftOut = report.blockerPhotos
    .filter((p) => !includedIds.has(p.id))
    .flatMap((p) => p.blockers);

  if (leftOut.length > 0) {
    const n = leftOut.length;
    const high = leftOut.filter((b) => b.severity === "High").length;
    const highNote = !high ? "" : n === 1 ? " (high severity)" : ` (${high} high)`;
    const noun = `${plural(n, "blocker")}${highNote}`;
    const verb = n === 1 ? (sent ? "wasn’t" : "isn’t") : sent ? "weren’t" : "aren’t";
    nudges.push({
      id: "left-out",
      tone: high ? "critical" : "warning",
      text:
        n === allFlagged.length
          ? `The drones flagged ${n} ${noun}. None ${sent ? "were" : "are"} in this email, so ${firstName} ${
              sent ? "was" : "will be"
            } told the home is approved.`
          : `${n} flagged ${noun} ${verb} in this email.`,
    });
  }

  return nudges;
}

export function LetterPane({
  member,
  firstName,
  report,
  placement,
  included,
  includedIds,
  draft,
  sentEmail,
  stampNow,
  phase,
  next,
  onRequestSend,
  onConfirmSend,
  onCancelSend,
}: {
  member: Member;
  firstName: string;
  report: DroneReport;
  placement: PlacementPhoto | null;
  included: BlockerPhoto[];
  includedIds: Set<string>;
  draft: MemberEmailDraft | null;
  sentEmail: SentEmail | null;
  stampNow: boolean;
  phase: SendPhase;
  next: Member | null;
  onRequestSend: () => void;
  onConfirmSend: () => void;
  onCancelSend: () => void;
}) {
  const approved = !includesBlockers(included);
  const fixes = collectFixes(included).length;
  const subject = approved ? SUBJECT_APPROVED : SUBJECT_ACTION_NEEDED;
  const sent = sentEmail !== null;
  const nudges = nudgesFor({ firstName, report, placement, includedIds, sent });

  const verdict = !placement ? (
    <>Choose where the battery goes to draft the email.</>
  ) : approved ? (
    <>
      {sent ? "Told" : "Tells"} {firstName} the home is approved.
    </>
  ) : (
    <>
      {sent ? "Asked" : "Asks"} {firstName} to fix {fixes} {plural(fixes, "thing")} before
      install.
    </>
  );

  return (
    <aside
      id="letter"
      aria-label={`Email to ${firstName}`}
      className="anim-rise xl:sticky xl:top-[88px] xl:self-start"
      style={delay(160)}
    >
      <div className="flex flex-col xl:max-h-[calc(100dvh-112px)]">
        <div className="flex items-baseline justify-between gap-4 border-t-2 border-ink pt-5">
          <h2 className="type-eyebrow text-ink">The email</h2>
          <p className="type-eyebrow text-ink-3">
            {sentEmail ? `Sent ${formatTime(sentEmail.sentAt)}` : "Live draft"}
          </p>
        </div>

        <div aria-live="polite" className="mt-5">
          <p
            key={`${Boolean(placement)}-${approved}-${fixes}-${sent}`}
            className="anim-fade flex items-start gap-3 text-[22px] font-semibold leading-[1.15] tracking-[-0.02em]"
          >
            <span className="mt-[3px] flex h-5 w-5 shrink-0 items-center justify-center">
              {!placement ? (
                <span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-ink-3" />
              ) : approved ? (
                <CheckGlyph className="h-5 w-5 text-good" strokeWidth={2.2} />
              ) : (
                <WarnGlyph className="h-5 w-5 text-warning" strokeWidth={2} />
              )}
            </span>
            <span>{verdict}</span>
          </p>
        </div>

        <div className="mt-5 flex items-center gap-4">
          <span className="type-eyebrow text-ink-3">Photos</span>
          <ul className="flex flex-wrap items-center gap-2" aria-label="Photos in this email">
            <li>
              {placement ? (
                <Thumb key={placement.id} attach="placement" src={placement.imageUrl} chosen />
              ) : (
                <span
                  data-attach="placement"
                  className="block h-9 w-12 rounded-[2px] border border-dashed border-rule-strong"
                >
                  <span className="sr-only">No placement photo yet</span>
                </span>
              )}
            </li>
            {included.map((photo) => (
              <li key={photo.id}>
                <Thumb attach={photo.id} src={photo.imageUrl} />
              </li>
            ))}
          </ul>
        </div>

        <div className="relative mt-5 flex min-h-0 flex-1 flex-col">
          <div
            className={`flex min-h-0 flex-1 flex-col overflow-hidden rounded-[3px] bg-sheet shadow-sheet ${
              stampNow ? "animate-[thud_460ms_ease-out_330ms_both]" : ""
            }`}
          >
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 border-b border-[#ececec] px-5 py-4 font-letter text-[12.5px] leading-[1.45]">
              <dt className="text-[#8a8a8a]">To</dt>
              <dd className="truncate text-[#27272a]">{member.email}</dd>
              <dt className="text-[#8a8a8a]">From</dt>
              <dd className="truncate text-[#27272a]">Base Power &lt;noreply@basepower.com&gt;</dd>
              <dt className="text-[#8a8a8a]">Subject</dt>
              <dd key={subject} className="anim-fade font-semibold text-[#18181b]">
                {subject}
              </dd>
            </dl>
            <div className="min-h-[200px] flex-1 overflow-y-auto overscroll-contain">
              {draft ? (
                <EmailBody html={draft.html} />
              ) : (
                <div className="px-6 py-8 font-letter text-[14px] leading-relaxed">
                  <p className="text-[#3f3f46]">Hi {firstName},</p>
                  <p className="mt-3 text-[#a1a1aa]">
                    The rest of this email writes itself once you choose where the battery goes.
                  </p>
                </div>
              )}
            </div>
          </div>
          {sentEmail ? <Postmark sentAt={sentEmail.sentAt} animate={stampNow} /> : null}
        </div>

        {nudges.length > 0 ? (
          <ul className="mt-5 space-y-2.5">
            {nudges.map((nudge) => (
              <li
                key={nudge.id}
                className="anim-fade flex items-start gap-2.5 text-[14px] leading-snug text-ink"
              >
                {nudge.tone === "critical" ? (
                  <CrossGlyph className="mt-[2px] h-4 w-4 shrink-0 text-critical" strokeWidth={2.2} />
                ) : (
                  <WarnGlyph className="mt-[1px] h-4 w-4 shrink-0 text-warning" strokeWidth={2} />
                )}
                <span>{nudge.text}</span>
              </li>
            ))}
          </ul>
        ) : null}

        {sentEmail ? (
          <div className="anim-fade mt-6" role="status">
            <p className="flex items-start gap-2 text-[15px] leading-snug">
              <CheckGlyph className="mt-[3px] h-4 w-4 shrink-0 text-good" strokeWidth={2.2} />
              <span>
                Sent to <span className="font-mono text-[13px]">{member.email}</span> at{" "}
                {formatTime(sentEmail.sentAt)}.
              </span>
            </p>
            {next ? (
              <Link
                href={`/members/${next.id}`}
                transitionTypes={["nav-forward"]}
                className="group mt-5 flex items-center justify-between gap-4 border-t border-rule pt-4"
              >
                <span>
                  <span className="type-eyebrow block text-ink-3">Next in queue</span>
                  <span className="text-[19px] font-semibold tracking-[-0.02em]">{next.name}</span>
                </span>
                <span aria-hidden className="text-[22px] transition-transform duration-500 ease-snap group-hover:translate-x-1.5">
                  →
                </span>
              </Link>
            ) : (
              <Link
                href="/"
                transitionTypes={["nav-back"]}
                className="group mt-5 flex items-center justify-between gap-4 border-t border-rule pt-4"
              >
                <span>
                  <span className="type-eyebrow block text-ink-3">Queue is clear</span>
                  <span className="text-[19px] font-semibold tracking-[-0.02em]">Back to the queue</span>
                </span>
                <span aria-hidden className="text-[22px] transition-transform duration-500 ease-snap group-hover:-translate-x-1.5">
                  ←
                </span>
              </Link>
            )}
          </div>
        ) : phase === "confirm" || phase === "sending" ? (
          <div className="anim-fade mt-6">
            <p className="text-[14px] leading-snug text-ink-2">
              This emails <span className="font-mono text-[13px] text-ink">{member.email}</span>{" "}
              now. A sent email can’t be unsent.
            </p>
            <div className="mt-3 flex gap-2">
              <PrimaryButton
                autoFocus
                busy={phase === "sending"}
                aria-disabled={phase === "sending"}
                onClick={phase === "sending" ? undefined : onConfirmSend}
                className="flex-1"
              >
                <span>{phase === "sending" ? "Sending…" : "Send now"}</span>
                <span aria-hidden>→</span>
              </PrimaryButton>
              <QuietButton onClick={onCancelSend} disabled={phase === "sending"}>
                Keep editing
              </QuietButton>
            </div>
          </div>
        ) : (
          <div className="mt-6">
            <PrimaryButton disabled={!draft} onClick={onRequestSend}>
              <span>Send to {firstName}</span>
              <span
                aria-hidden
                className="transition-transform duration-500 ease-snap group-hover/btn:translate-x-1"
              >
                →
              </span>
            </PrimaryButton>
            {!draft ? (
              <p className="mt-2.5 text-[13px] text-ink-3">Choose a placement in step 01 first.</p>
            ) : null}
            {phase === "error" ? (
              <p role="alert" className="mt-3 flex items-start gap-2 text-[14px] text-critical">
                <CrossGlyph className="mt-[2px] h-4 w-4 shrink-0" strokeWidth={2.2} />
                The email didn’t send, and nothing went out. Try again.
              </p>
            ) : null}
          </div>
        )}
      </div>
    </aside>
  );
}
