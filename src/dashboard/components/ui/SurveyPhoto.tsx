"use client";

import { severityColor } from "@/components/ui/glyphs";
import type { BlockerSeverity, Box } from "@/lib/types";
import Image from "next/image";
import type { CSSProperties, ReactNode } from "react";

export interface Detection {
  id: string;
  n: number;
  box: Box;
  severity: BlockerSeverity;
}

export interface Proposal {
  box: Box;
  label: string;
}

const badgeText: Record<BlockerSeverity, string> = {
  High: "#fff",
  Medium: "var(--color-ink)",
  Low: "#fff",
};

function boxStyle(box: Box): CSSProperties {
  return {
    left: `${box.x}%`,
    top: `${box.y}%`,
    width: `${box.width}%`,
    height: `${box.height}%`,
  };
}

/**
 * A drone photo with its annotations. Dashed blue = a proposed footprint
 * (not built yet). Brackets = something a drone flagged, numbered to match the
 * list beside it.
 */
export function SurveyPhoto({
  src,
  alt,
  detections = [],
  proposals = [],
  activeId = null,
  onActivate,
  eager = false,
  sizes = "(min-width: 1280px) 420px, (min-width: 640px) 50vw, 100vw",
  className = "",
  children,
}: {
  src: string;
  alt: string;
  detections?: Detection[];
  proposals?: Proposal[];
  activeId?: string | null;
  onActivate?: (id: string | null) => void;
  /** Above-the-fold photos load immediately (`priority` is deprecated in Next 16). */
  eager?: boolean;
  sizes?: string;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <div
      className={`relative aspect-[4/3] overflow-hidden rounded-[3px] bg-paper-2 ${className}`}
    >
      <Image
        src={src}
        alt={alt}
        fill
        unoptimized
        loading={eager ? "eager" : "lazy"}
        fetchPriority={eager ? "high" : "auto"}
        sizes={sizes}
        className="object-cover"
      />

      {proposals.map((proposal, i) => (
        <div
          key={i}
          className="absolute border-[1.5px] border-dashed border-signal bg-signal/[0.08]"
          style={boxStyle(proposal.box)}
        >
          <span className="absolute -left-[1.5px] top-0 -translate-y-full bg-signal px-1.5 font-mono text-[10px] font-medium uppercase leading-[16px] tracking-[0.08em] text-white">
            {proposal.label}
          </span>
        </div>
      ))}

      {detections.map((d) => {
        const active = activeId === d.id;
        return (
          <div
            key={d.id}
            className="detection absolute"
            data-active={active || undefined}
            data-dim={(activeId !== null && !active) || undefined}
            style={{ ...boxStyle(d.box), "--c": severityColor[d.severity] } as CSSProperties}
            onMouseEnter={() => onActivate?.(d.id)}
            onMouseLeave={() => onActivate?.(null)}
          >
            <span
              className="absolute left-0 top-0 flex h-[18px] min-w-[18px] -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full px-1 font-mono text-[11px] font-semibold ring-2 ring-white/80"
              style={{ background: severityColor[d.severity], color: badgeText[d.severity] }}
            >
              {d.n}
            </span>
          </div>
        );
      })}

      {children}
    </div>
  );
}

export function Viewfinder({
  active,
  children,
  className = "",
}: {
  active: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`vf relative ${className}`} data-active={active || undefined}>
      {children}
      {(["tl", "tr", "bl", "br"] as const).map((corner) => (
        <span key={corner} data-c={corner} className="vf-corner" aria-hidden />
      ))}
    </div>
  );
}
