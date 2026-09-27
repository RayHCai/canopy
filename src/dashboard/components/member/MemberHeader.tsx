"use client";

import { CheckGlyph } from "@/components/ui/glyphs";
import { formatTime } from "@/lib/format";
import { hubspotContactSearchUrl, siteSurveyReviewSlackUrl } from "@/lib/links";
import { stageOf } from "@/lib/memberSession";
import type { Member, SentEmail } from "@/lib/types";
import Link from "next/link";
import { ViewTransition, type ReactNode } from "react";

function StageLine({ member, sentEmail }: { member: Member; sentEmail: SentEmail | null }) {
  const stage = stageOf[member.reportStatus];
  return (
    <p key={stage} className="type-eyebrow anim-fade flex items-center gap-2 text-ink-2">
      {stage === "sent" ? (
        <>
          <CheckGlyph className="h-3.5 w-3.5 text-good" strokeWidth={2.2} />
          Sent{sentEmail ? ` ${formatTime(sentEmail.sentAt)}` : ""}
        </>
      ) : stage === "ready" ? (
        <>
          <span className="h-2 w-2 rounded-full bg-ink" />
          Ready for review
        </>
      ) : (
        <>
          <span className="h-2 w-2 rounded-full border border-dashed border-ink-3" />
          Waiting on drones
        </>
      )}
    </p>
  );
}

function OutLink({
  href,
  envVar,
  children,
}: {
  href: string | null;
  envVar: string;
  children: ReactNode;
}) {
  const shape =
    "inline-flex h-9 items-center gap-2 rounded-full border px-3.5 text-[14px] font-medium";

  if (!href) {
    return (
      <span
        aria-disabled="true"
        title={`Not linked yet: set ${envVar} in .env.local`}
        className={`${shape} cursor-not-allowed border-dashed border-rule-strong text-ink-3`}
      >
        {children}
      </span>
    );
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`${shape} group border-rule-strong text-ink transition-colors duration-200 hover:border-ink hover:bg-paper-2`}
    >
      {children}
      <span
        aria-hidden
        className="text-ink-3 transition-[translate,color] duration-300 ease-snap group-hover:-translate-y-px group-hover:translate-x-px group-hover:text-ink"
      >
        ↗
      </span>
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
}

export function MemberHeader({
  member,
  sentEmail,
}: {
  member: Member;
  sentEmail: SentEmail | null;
}) {
  return (
    <header className="pt-8 sm:pt-10">
      <Link
        href="/"
        transitionTypes={["nav-back"]}
        className="type-eyebrow group inline-flex items-center gap-2 text-ink-3 transition-colors hover:text-ink"
      >
        <span aria-hidden className="transition-transform duration-300 ease-snap group-hover:-translate-x-1">
          ←
        </span>
        Queue
      </Link>

      <div className="mt-8 flex flex-wrap items-end justify-between gap-x-10 gap-y-5">
        <div className="min-w-0">
          <StageLine member={member} sentEmail={sentEmail} />
          {/* One line where it fits; stacked below that, so no separator is left dangling. */}
          <div className="mt-3 flex flex-col gap-y-0.5 text-[18px] leading-snug lg:flex-row lg:flex-wrap lg:items-baseline lg:gap-x-3">
            <ViewTransition name={`member-name-${member.id}`} share="morph" default="none">
              <h1 className="w-fit font-semibold tracking-[-0.015em]">{member.name}</h1>
            </ViewTransition>
            <span aria-hidden className="hidden text-rule-strong lg:inline">
              ·
            </span>
            <p className="text-ink-2">{member.address}</p>
            <span aria-hidden className="hidden text-rule-strong lg:inline">
              ·
            </span>
            <p className="text-ink-2">{member.email}</p>
          </div>
        </div>

        <nav aria-label="Open elsewhere" className="flex flex-wrap items-center gap-2">
          <OutLink
            href={hubspotContactSearchUrl(member.email)}
            envVar="NEXT_PUBLIC_HUBSPOT_PORTAL_ID"
          >
            HubSpot
          </OutLink>
          <OutLink
            href={siteSurveyReviewSlackUrl}
            envVar="NEXT_PUBLIC_SLACK_SITE_SURVEY_REVIEW_URL"
          >
            Slack · Site survey review
          </OutLink>
        </nav>
      </div>
    </header>
  );
}
