import { Today } from "@/components/Today";
import { CanopyMark } from "@/components/ui/glyphs";
import Link from "next/link";
import type { ReactNode } from "react";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-dvh">
      <header
        className="sticky top-0 z-40 bg-paper/92 backdrop-blur-md"
        style={{ viewTransitionName: "masthead" }}
      >
        <div className="mx-auto flex h-16 max-w-[1320px] items-center gap-5 px-5 sm:px-10">
          <Link
            href="/"
            transitionTypes={["nav-back"]}
            className="flex items-center gap-2 rounded-[2px]"
          >
            <CanopyMark className="h-[22px] w-[22px]" />
            <span className="text-[15px] font-extrabold uppercase tracking-[0.1em] [font-stretch:125%]">
              Canopy
            </span>
          </Link>
          <span className="type-eyebrow hidden text-ink-3 sm:inline">
            Site survey review
          </span>
          <Today className="type-eyebrow ml-auto text-ink-3" />
        </div>
        <div className="masthead-rule h-px bg-rule" />
      </header>
      <main className="mx-auto max-w-[1320px] px-5 pb-40 sm:px-10">{children}</main>
    </div>
  );
}
