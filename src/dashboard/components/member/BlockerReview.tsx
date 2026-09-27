"use client";

import { CheckGlyph, SeverityTag, severityColor } from "@/components/ui/glyphs";
import { SurveyPhoto, Viewfinder } from "@/components/ui/SurveyPhoto";
import { Tags } from "@/components/ui/Tags";
import { blockerTags, bySeverity, plural } from "@/lib/format";
import type { BlockerPhoto, BlockerSeverity } from "@/lib/types";
import {
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";

function NumberBadge({ n, severity }: { n: number; severity: BlockerSeverity }) {
  return (
    <span
      aria-hidden
      className="flex h-[22px] min-w-[22px] items-center justify-center rounded-full px-1 font-mono text-[12px] font-semibold"
      style={{
        background: severityColor[severity],
        color: severity === "Medium" ? "var(--color-ink)" : "#fff",
      }}
    >
      {n}
    </span>
  );
}

function IncludeSwitch({
  id,
  checked,
  disabled,
  onChange,
}: {
  id: string;
  checked: boolean;
  disabled: boolean;
  onChange: () => void;
}) {
  return (
    <label
      className={`group/sw inline-flex items-center gap-3 self-start ${
        disabled ? "cursor-default" : "cursor-pointer"
      }`}
    >
      <input
        id={id}
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        onChange={onChange}
        className="peer sr-only"
      />
      <span
        aria-hidden
        className={`relative h-[30px] w-[52px] shrink-0 rounded-full transition-colors duration-300 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal ${
          checked ? "bg-signal" : "bg-rule-strong group-hover/sw:bg-ink-3"
        }`}
      >
        <span
          className={`absolute left-[3px] top-[3px] h-6 w-6 rounded-full bg-white shadow-[0_1px_3px_rgb(22_21_15/0.3)] transition-transform duration-500 ease-spring ${
            checked ? "translate-x-[22px]" : ""
          }`}
        />
      </span>
      <span className="text-[16px] font-medium">
        {checked ? "Included in the email" : "Include in the email"}
      </span>
    </label>
  );
}

function BlockerFigure({
  photo,
  startAt,
  included,
  locked,
  onToggle,
}: {
  photo: BlockerPhoto;
  startAt: number;
  included: boolean;
  locked: boolean;
  onToggle: (photoEl: HTMLElement | null) => void;
}) {
  const blockers = useMemo(() => bySeverity(photo.blockers), [photo.blockers]);
  const switchId = useId();
  const [activeId, setActiveId] = useState<string | null>(null);
  const [inView, setInView] = useState(false);
  const figureRef = useRef<HTMLElement>(null);
  const photoRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLLIElement | null)[]>([]);
  const pathRefs = useRef<(SVGPathElement | null)[]>([]);
  const dotRefs = useRef<(SVGCircleElement | null)[]>([]);

  // Leader lines run from each bracket to its entry in the list. They are
  // measured, not guessed, so they stay attached at any width.
  useLayoutEffect(() => {
    const figure = figureRef.current;
    const photoEl = photoRef.current;
    if (!figure || !photoEl) return;

    const draw = () => {
      const f = figure.getBoundingClientRect();
      const p = photoEl.getBoundingClientRect();
      blockers.forEach((blocker, i) => {
        const path = pathRefs.current[i];
        const dot = dotRefs.current[i];
        const item = itemRefs.current[i];
        if (!path || !dot || !item) return;
        const r = item.getBoundingClientRect();
        // Stacked layout: the list sits under the photo and the numbers do the linking.
        const stacked = r.left < p.right;
        dot.style.display = stacked ? "none" : "";
        if (stacked) {
          path.setAttribute("d", "");
          return;
        }
        const { box } = blocker;
        const x1 = p.left - f.left + ((box.x + box.width) / 100) * p.width;
        const y1 = p.top - f.top + ((box.y + box.height / 2) / 100) * p.height;
        const xm = p.right - f.left + 20 + i * 8;
        const x2 = r.left - f.left - 14;
        const y2 = r.top - f.top + 11;
        path.setAttribute("d", `M${x1} ${y1} H${xm} V${y2} H${x2}`);
        dot.setAttribute("cx", String(x1));
        dot.setAttribute("cy", String(y1));
      });
    };

    draw();
    // Web fonts can reflow the list without resizing the figure.
    document.fonts?.ready.then(draw);
    const observer = new ResizeObserver(draw);
    observer.observe(figure);
    return () => observer.disconnect();
  }, [blockers]);

  useEffect(() => {
    const figure = figureRef.current;
    if (!figure) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(figure);
    return () => observer.disconnect();
  }, []);

  const activeSeverity = blockers.find((b) => b.id === activeId)?.severity;

  return (
    <figure
      ref={figureRef}
      className="relative grid gap-x-16 gap-y-8 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]"
    >
      {/* The photo is a label for the switch, so clicking it includes it, like choosing a placement. */}
      <label
        htmlFor={switchId}
        data-locked={locked || undefined}
        className={`vf-aim block ${locked ? "cursor-default" : "cursor-pointer"}`}
      >
        <Viewfinder active={included}>
          <div ref={photoRef} data-photo>
            <SurveyPhoto
              src={photo.imageUrl}
              alt={`Survey photo with ${blockers.length} flagged ${plural(blockers.length, "blocker")}`}
              detections={blockers.map((b, i) => ({
                id: b.id,
                n: startAt + i,
                box: b.box,
                severity: b.severity,
              }))}
              activeId={activeId}
              onActivate={setActiveId}
            />
          </div>
        </Viewfinder>
      </label>

      <figcaption className="flex min-w-0 flex-col">
        <ol className="space-y-8">
          {blockers.map((blocker, i) => (
            <li
              key={blocker.id}
              ref={(el) => {
                itemRefs.current[i] = el;
              }}
              onMouseEnter={() => setActiveId(blocker.id)}
              onMouseLeave={() => setActiveId(null)}
              className={`transition-opacity duration-300 ${
                activeId && activeId !== blocker.id ? "opacity-40" : ""
              }`}
            >
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <NumberBadge n={startAt + i} severity={blocker.severity} />
                <span className="text-[19px] font-semibold leading-tight tracking-[-0.015em]">
                  {blocker.type}
                </span>
                <SeverityTag severity={blocker.severity} />
              </div>
              <p className="mt-2 font-mono text-[12.5px] leading-relaxed text-ink-3">
                {blocker.description}
              </p>
              <p
                className={`mt-3 border-l-2 pl-3 text-[15px] leading-snug transition-colors duration-500 ${
                  included ? "border-signal text-ink" : "border-rule-strong text-ink-2"
                }`}
              >
                <span className="type-eyebrow block pb-0.5 text-ink-3">
                  {included ? "Fix · in the email" : "Fix"}
                </span>
                {blocker.requiredFix}
              </p>
            </li>
          ))}
        </ol>
        <div className="mt-10">
          <IncludeSwitch
            id={switchId}
            checked={included}
            disabled={locked}
            onChange={() => onToggle(photoRef.current)}
          />
        </div>
      </figcaption>

      <svg
        aria-hidden
        className="pointer-events-none absolute inset-0 hidden h-full w-full overflow-visible md:block"
        data-inview={inView || undefined}
      >
        {blockers.map((blocker, i) => {
          const isActive = activeId === blocker.id;
          const color =
            isActive && activeSeverity ? severityColor[activeSeverity] : "var(--color-ink-3)";
          return (
            <g
              key={blocker.id}
              className="transition-opacity duration-300"
              style={{ opacity: activeId && !isActive ? 0.2 : 1 }}
            >
              <path
                ref={(el) => {
                  pathRefs.current[i] = el;
                }}
                pathLength={1}
                fill="none"
                stroke={color}
                strokeWidth={isActive ? 1.75 : 1}
                className="leader transition-[stroke,stroke-width] duration-300"
                style={{ "--i": i } as CSSProperties}
              />
              <circle
                ref={(el) => {
                  dotRefs.current[i] = el;
                }}
                r={3}
                fill={color}
                className="leader-dot transition-[fill] duration-300"
                style={{ "--i": i } as CSSProperties}
              />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}

export function BlockerReview({
  firstName,
  photos,
  includedIds,
  locked,
  onToggle,
}: {
  firstName: string;
  photos: BlockerPhoto[];
  includedIds: Set<string>;
  locked: boolean;
  onToggle: (photo: BlockerPhoto, photoEl: HTMLElement | null) => void;
}) {
  if (photos.length === 0) {
    return (
      <div className="flex items-start gap-4">
        <CheckGlyph className="mt-1 h-6 w-6 shrink-0 text-good" strokeWidth={2.2} />
        <div>
          <p className="text-[22px] font-semibold leading-tight tracking-[-0.02em]">
            The drones didn’t flag anything.
          </p>
          <p className="mt-2 max-w-[56ch] text-[16px] leading-relaxed text-ink-2">
            No blockers were found in this survey, so the email will tell {firstName} the home is
            approved.
          </p>
        </div>
      </div>
    );
  }

  // Numbering runs across photos, so "blocker 3" means one thing on this page.
  const startsAt = photos.map(
    (_, i) => 1 + photos.slice(0, i).reduce((sum, p) => sum + p.blockers.length, 0),
  );

  return (
    <div>
      <Tags tags={blockerTags(photos.flatMap((p) => p.blockers))} label="Blocker work" />
      <div className="mt-12 space-y-24">
        {photos.map((photo, i) => (
          <BlockerFigure
            key={photo.id}
            photo={photo}
            startAt={startsAt[i]}
            included={includedIds.has(photo.id)}
            locked={locked}
            onToggle={(photoEl) => onToggle(photo, photoEl)}
          />
        ))}
      </div>
    </div>
  );
}
