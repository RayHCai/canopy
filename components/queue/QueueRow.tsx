"use client";

import { CheckGlyph, SeverityMark } from "@/components/ui/glyphs";
import { Tags } from "@/components/ui/Tags";
import { formatTime, plural } from "@/lib/format";
import type { QueueRow as Row } from "@/lib/memberSession";
import Image from "next/image";
import Link from "next/link";
import { ViewTransition, type CSSProperties, type ReactNode } from "react";

function Finding({ row, className = "" }: { row: Row; className?: string }) {
  let mark: ReactNode;
  let title: string;
  let detail: string;

  if (row.stage === "waiting") {
    mark = <span className="h-3 w-3 rounded-full border-[1.5px] border-dashed border-ink-3" />;
    title = "No report yet";
    detail = "Waiting on drones";
  } else if (row.stage === "sent") {
    mark = <CheckGlyph className="h-4 w-4 text-good" />;
    title = row.sent ? `Sent ${formatTime(row.sent.sentAt)}` : "Sent";
    detail = row.sent
      ? row.sent.approved
        ? "Approval email"
        : "Action-needed email"
      : "Email sent";
  } else if (row.blockers.worst) {
    const { total } = row.blockers;
    mark = <SeverityMark severity={row.blockers.worst} className="h-4 w-4" />;
    title = `${total} ${plural(total, "blocker")} found`;
    detail = (["High", "Medium", "Low"] as const)
      .filter((s) => row.blockers[s] > 0)
      .map((s) => `${row.blockers[s]} ${s.toLowerCase()}`)
      .join(" · ");
  } else {
    mark = <CheckGlyph className="h-4 w-4 text-good" />;
    title = "No blockers found";
    detail = `${row.photos.length} ${plural(row.photos.length, "photo")} to review`;
  }

  return (
    <div className={`flex items-center gap-3.5 ${className}`}>
      <span className="flex h-5 w-5 shrink-0 items-center justify-center">{mark}</span>
      <div className="min-w-0">
        <p className="text-[16px] font-medium leading-tight">{title}</p>
        <p className="type-eyebrow mt-1 text-ink-3">{detail}</p>
      </div>
    </div>
  );
}

/** The survey's actual photos, stacked; they fan out when the row is hovered. */
function PhotoStack({ photos, className = "" }: { photos: Row["photos"]; className?: string }) {
  const shown = photos.slice(0, 4);
  return (
    <div className={`relative h-[54px] w-[184px] ${className}`} aria-hidden>
      {shown.map((photo, i) => (
        <span
          key={photo.id}
          className="absolute top-0 block h-[54px] w-[72px] overflow-hidden rounded-[3px] shadow-[0_1px_3px_rgb(22_21_15/0.18)] ring-1 ring-ink/10 transition-[left,rotate] duration-500 ease-snap [left:calc(var(--i)*22px)] [rotate:calc((var(--i)-var(--mid))*3deg)] group-hover:[left:calc(var(--i)*37px)] group-hover:[rotate:0deg]"
          style={
            {
              "--i": i,
              "--mid": (shown.length - 1) / 2,
              zIndex: i,
              transitionDelay: `${i * 30}ms`,
            } as CSSProperties
          }
        >
          <Image
            src={photo.imageUrl}
            alt=""
            width={72}
            height={54}
            unoptimized
            className="h-full w-full object-cover"
          />
        </span>
      ))}
    </div>
  );
}

export function QueueRow({ row }: { row: Row }) {
  return (
    <Link
      href={`/members/${row.id}`}
      transitionTypes={["nav-forward"]}
      className="group grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-4 border-t border-rule py-6 [outline-offset:-2px] md:grid-cols-[minmax(0,1.1fr)_minmax(210px,0.8fr)_auto_auto] md:gap-x-10"
    >
      <div className="min-w-0">
        <ViewTransition name={`member-name-${row.id}`} share="morph" default="none">
          <span className="block w-fit max-w-full text-[clamp(24px,2.4vw,32px)] font-semibold leading-[1.2] tracking-[-0.028em] [font-stretch:104%]">
            <span className="bg-[linear-gradient(currentColor,currentColor)] bg-[length:0%_2px] bg-[position:0_100%] bg-no-repeat transition-[background-size] duration-500 ease-snap group-hover:bg-[length:100%_2px] group-focus-visible:bg-[length:100%_2px]">
              {row.name}
            </span>
          </span>
        </ViewTransition>
        <span className="mt-1 block truncate text-[14px] text-ink-3">{row.address}</span>
        <Tags tags={row.tags} label="Blocker work" className="mt-3" />
      </div>

      <Finding row={row} className="order-3 col-span-2 md:order-none md:col-span-1" />
      <PhotoStack photos={row.photos} className="hidden lg:block" />

      <span
        aria-hidden
        className="text-[26px] leading-none text-ink-3 transition-[translate,color] duration-500 ease-snap group-hover:translate-x-2 group-hover:text-ink"
      >
        →
      </span>
    </Link>
  );
}
