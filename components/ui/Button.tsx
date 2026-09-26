"use client";

import type { ButtonHTMLAttributes } from "react";

/** Ink at rest; the signal color sweeps in on hover because pressing it is a decision. */
export function PrimaryButton({
  busy = false,
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { busy?: boolean }) {
  return (
    <button
      type="button"
      aria-busy={busy || undefined}
      className={`group/btn relative isolate flex h-14 w-full items-center justify-between gap-4 overflow-hidden rounded-[3px] bg-ink px-5 text-left text-[16px] font-semibold tracking-[-0.01em] text-white transition-[background-color,color,scale] duration-300 ease-snap active:scale-[0.985] disabled:cursor-not-allowed disabled:bg-paper-2 disabled:text-ink-3 ${className}`}
      {...rest}
    >
      <span
        aria-hidden
        className="absolute inset-0 -z-10 origin-left scale-x-0 bg-signal transition-transform duration-500 ease-snap group-hover/btn:scale-x-100 group-disabled/btn:hidden"
      />
      {busy ? (
        <span aria-hidden className="absolute inset-x-0 bottom-0 h-[3px] overflow-hidden">
          <span className="block h-full w-1/3 animate-[sweep_1.2s_var(--ease-swift)_infinite] bg-white/80" />
        </span>
      ) : null}
      {children}
    </button>
  );
}

export function QuietButton({
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`inline-flex h-14 items-center justify-center rounded-[3px] px-5 text-[15px] font-medium text-ink-2 underline decoration-rule-strong underline-offset-4 transition-colors duration-200 hover:text-ink hover:decoration-ink disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
