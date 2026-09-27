"use client";

import { useSyncExternalStore } from "react";

const subscribe = (onChange: () => void) => {
  const id = setInterval(onChange, 60_000);
  return () => clearInterval(id);
};

const readToday = () =>
  new Date().toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
  });

/** Rendered only in the browser: a prerendered date would be the build's date, not today's. */
export function Today({ className = "" }: { className?: string }) {
  const today = useSyncExternalStore(subscribe, readToday, () => null);
  return <span className={className}>{today}</span>;
}
