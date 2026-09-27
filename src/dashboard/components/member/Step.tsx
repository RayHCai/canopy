import { CheckGlyph } from "@/components/ui/glyphs";
import type { ReactNode } from "react";

export function Step({
  n,
  title,
  meta,
  done = false,
  children,
}: {
  n: string;
  title: string;
  meta: string;
  done?: boolean;
  children: ReactNode;
}) {
  const id = `step-${n}`;
  return (
    <section aria-labelledby={id}>
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-2 border-t-2 border-ink pt-5">
        <span aria-hidden className="relative inline-flex w-7 font-mono text-[13px] text-ink-3">
          <span
            className={`transition-[opacity,translate] duration-500 ease-snap ${
              done ? "-translate-y-2 opacity-0" : ""
            }`}
          >
            {n}
          </span>
          <CheckGlyph
            strokeWidth={2.4}
            className={`absolute left-0 top-px h-4 w-4 text-signal transition-[opacity,scale] duration-500 ease-spring ${
              done ? "scale-100 opacity-100" : "scale-50 opacity-0"
            }`}
          />
        </span>
        <h2
          id={id}
          className="text-[clamp(24px,2.5vw,34px)] font-semibold leading-[1.1] tracking-[-0.028em]"
        >
          {title}
        </h2>
        <p className="type-eyebrow ml-auto text-ink-3">{meta}</p>
      </div>
      <div className="mt-12">{children}</div>
    </section>
  );
}
