import { severityRank } from "@/lib/format";
import type { BlockerSeverity } from "@/lib/types";
import type { SVGProps } from "react";

type GlyphProps = SVGProps<SVGSVGElement>;

const stroke = {
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
} as const;

export function CheckGlyph(props: GlyphProps) {
  return (
    <svg {...stroke} {...props}>
      <path d="M3 8.5 6.2 11.5 13 4.5" />
    </svg>
  );
}

export function CrossGlyph(props: GlyphProps) {
  return (
    <svg {...stroke} {...props}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

export function WarnGlyph(props: GlyphProps) {
  return (
    <svg {...stroke} {...props}>
      <path d="M8 2.2 14.2 13.2H1.8Z" />
      <path d="M8 6.4v3.1M8 11.3v.2" />
    </svg>
  );
}

export function SearchGlyph(props: GlyphProps) {
  return (
    <svg {...stroke} {...props}>
      <circle cx="7" cy="7" r="4.5" />
      <path d="m10.5 10.5 3.5 3.5" />
    </svg>
  );
}

/** A dome over the ground line: the canopy, seen side-on. */
export function CanopyMark(props: GlyphProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M3 16a9 9 0 0 1 18 0Z" fill="currentColor" />
      <rect x="3" y="18.5" width="18" height="2.5" fill="currentColor" />
    </svg>
  );
}

export const severityColor: Record<BlockerSeverity, string> = {
  High: "var(--color-critical)",
  Medium: "var(--color-warning)",
  Low: "var(--color-ink-3)",
};

/**
 * Severity as a level meter: the more bars filled, the worse it is.
 * Shape carries the order, so it still reads without color.
 */
export function SeverityMark({
  severity,
  className = "h-3 w-3",
}: {
  severity: BlockerSeverity;
  className?: string;
}) {
  const level = severityRank[severity];
  return (
    <svg viewBox="0 0 12 12" className={className} aria-hidden>
      {[0, 1, 2].map((i) => (
        <rect
          key={i}
          x="0"
          y={9.25 - i * 4.25}
          width="12"
          height="2.75"
          rx="0.6"
          fill={i < level ? severityColor[severity] : "var(--color-rule-strong)"}
        />
      ))}
    </svg>
  );
}

export function SeverityTag({ severity }: { severity: BlockerSeverity }) {
  return (
    <span className="type-eyebrow inline-flex items-center gap-1.5 text-ink-2">
      <SeverityMark severity={severity} />
      {severity}
    </span>
  );
}
