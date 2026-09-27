"use client";

import { StatusNotice } from "@/components/StatusNotice";
import { QueueRow } from "@/components/queue/QueueRow";
import { SearchGlyph } from "@/components/ui/glyphs";
import { delay } from "@/components/ui/motion";
import { useQueue, type QueueStage } from "@/lib/memberSession";
import { useEffect, useRef, useState } from "react";

const groups: { stage: QueueStage; title: string; empty: string }[] = [
  { stage: "ready", title: "Ready for review", empty: "Nothing is waiting on you." },
  { stage: "sent", title: "Sent", empty: "Nothing sent yet." },
];

export function QueueClient() {
  const { rows, loading, error } = useQueue();
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

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

  if (loading) {
    return (
      <div role="status" className="anim-fade pt-12 sm:pt-16">
        <p className="type-eyebrow text-ink-3">Loading</p>
        <div className="mt-4 h-[2px] overflow-hidden bg-rule">
          <div className="h-full w-1/3 animate-[sweep_1.3s_var(--ease-swift)_infinite] bg-ink" />
        </div>
      </div>
    );
  }

  // A failed poll keeps the last good list on screen; only a queue with
  // nothing to fall back to shows the error.
  if (error && rows.length === 0) {
    return (
      <div className="pt-12 sm:pt-16">
        <StatusNotice
          eyebrow="Review API offline"
          title="Can’t reach the review API."
          message={error}
        />
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="pt-12 sm:pt-16">
        <StatusNotice
          eyebrow="Nothing to review"
          title="No surveys yet."
          message="Run canopy-view and complete intake to send a swarm out. Its review will show up here."
        />
      </div>
    );
  }

  return (
    <>
      {/* The search is the page's masthead: it spans the rule the queue hangs from. */}
      <label className="anim-rise group flex cursor-text items-center gap-4 border-b-2 border-ink pb-4 pt-12 transition-colors duration-300 has-[input:focus]:border-signal sm:pt-16">
        <SearchGlyph className="h-5 w-5 shrink-0 text-ink-3 transition-colors group-has-[input:focus]:text-signal" />
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
          placeholder="Find a member by name, address or email"
          aria-label="Find a member by name, address or email"
          className="min-w-0 flex-1 bg-transparent py-1 text-[clamp(20px,1.9vw,26px)] font-medium tracking-[-0.02em] outline-none placeholder:font-normal placeholder:text-ink-3 [&::-webkit-search-cancel-button]:hidden"
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
      </label>

      {q && matches.length === 0 ? (
        <p className="anim-fade pt-12 text-[20px] text-ink-2">No one matches “{query.trim()}”.</p>
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
