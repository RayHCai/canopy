"use client";

import { QueueRow } from "@/components/queue/QueueRow";
import { SearchGlyph } from "@/components/ui/glyphs";
import { delay } from "@/components/ui/motion";
import { RollingNumber } from "@/components/ui/RollingNumber";
import { plural } from "@/lib/format";
import { useQueue, type QueueStage } from "@/lib/memberSession";
import { useEffect, useRef, useState } from "react";

/**
 * The ready count as of the last visit to the queue, so a count that changed
 * while you were on a member page rolls to its new value instead of just being different.
 */
let lastSeenReady: number | null = null;

const groups: { stage: QueueStage; title: string; empty: string }[] = [
  { stage: "ready", title: "Ready for review", empty: "Nothing is waiting on you." },
  { stage: "sent", title: "Sent", empty: "Nothing sent yet." },
];

export function QueueClient() {
  const { rows, readyCount } = useQueue();
  const [from] = useState(() => lastSeenReady);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    lastSeenReady = readyCount;
  }, [readyCount]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && e.target.closest("input, textarea");
      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const q = query.trim().toLowerCase();
  const matches = rows.filter(
    (r) =>
      groups.some((g) => g.stage === r.stage) &&
      (!q ||
        r.name.toLowerCase().includes(q) ||
        r.address.toLowerCase().includes(q) ||
        r.email.toLowerCase().includes(q)),
  );

  return (
    <>
      <div className="anim-rise flex flex-wrap items-end justify-between gap-x-12 gap-y-6 border-b-2 border-ink pb-4 pt-12 transition-colors duration-300 has-[input:focus]:border-signal sm:pt-16">
        <div>
          <p className="type-eyebrow text-ink-3">Review queue</p>
          <h1 className="mt-3 text-[clamp(26px,2.4vw,32px)] font-semibold leading-[1.1] tracking-[-0.025em]">
            {readyCount > 0 ? (
              <>
                <RollingNumber value={readyCount} from={from ?? undefined} />{" "}
                {plural(readyCount, "home")} ready for review
              </>
            ) : (
              "Nothing to review"
            )}
          </h1>
        </div>

        <div className="flex w-full items-center gap-3 sm:w-[340px]">
          <SearchGlyph className="h-4 w-4 shrink-0 text-ink-3" />
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setQuery("");
                e.currentTarget.blur();
              }
            }}
            placeholder="Find a member"
            aria-label="Find a member by name, address or email"
            className="min-w-0 flex-1 bg-transparent py-1.5 text-[16px] outline-none placeholder:text-ink-3 [&::-webkit-search-cancel-button]:hidden"
          />
          {query ? (
            <button
              type="button"
              onClick={() => {
                setQuery("");
                inputRef.current?.focus();
              }}
              className="type-eyebrow text-ink-3 transition-colors hover:text-ink"
            >
              Clear
            </button>
          ) : (
            <kbd className="hidden rounded-[3px] border border-rule-strong px-1.5 font-mono text-[11px] leading-5 text-ink-3 sm:block">
              /
            </kbd>
          )}
        </div>
      </div>

      {q && matches.length === 0 ? (
        <p className="anim-fade pt-12 text-[20px] text-ink-2">
          No one matches “{query.trim()}”.
        </p>
      ) : null}

      {groups.map((group, gi) => {
        const inGroup = matches.filter((r) => r.stage === group.stage);
        if (q && inGroup.length === 0) return null;
        return (
          <section
            key={group.stage}
            aria-labelledby={`queue-${group.stage}`}
            className="anim-rise mt-12"
            style={delay(120 + gi * 90)}
          >
            <h2 id={`queue-${group.stage}`} className="flex items-baseline gap-3 pb-4">
              <span className="type-eyebrow text-ink">{group.title}</span>
              <span className="type-eyebrow tabular-nums text-ink-3">
                {String(inGroup.length).padStart(2, "0")}
              </span>
            </h2>
            {inGroup.length === 0 ? (
              <p className="border-t border-rule py-6 text-[16px] text-ink-3">{group.empty}</p>
            ) : (
              <ul>
                {inGroup.map((row) => (
                  <li key={row.id} className="anim-fade">
                    <QueueRow row={row} />
                  </li>
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </>
  );
}
