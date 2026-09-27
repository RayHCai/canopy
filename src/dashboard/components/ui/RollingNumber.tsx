"use client";

import { useEffect, useState } from "react";

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/**
 * Digits roll like an odometer from `from` to `value`, so a count that changed
 * while you were away visibly changes when you come back.
 */
export function RollingNumber({
  value,
  from = value,
}: {
  value: number;
  from?: number;
}) {
  const [shown, setShown] = useState(from);

  useEffect(() => {
    let inner = 0;
    // Two frames: let the starting digits paint before rolling to the new ones.
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setShown(value));
    });
    return () => {
      cancelAnimationFrame(outer);
      cancelAnimationFrame(inner);
    };
  }, [value]);

  const digits = String(Math.max(0, shown)).split("").map(Number);

  return (
    <span className="inline-flex tabular-nums">
      <span className="sr-only">{value}</span>
      {digits.map((digit, i) => (
        <span
          key={digits.length - i}
          aria-hidden
          className="relative inline-block leading-none [clip-path:inset(0_-0.2em)]"
        >
          <span className="invisible">0</span>
          <span
            className="absolute inset-x-0 top-0 flex flex-col transition-transform duration-[1100ms] ease-spring"
            style={{ transform: `translateY(${-digit}em)` }}
          >
            {DIGITS.map((n) => (
              <span key={n} className="block h-[1em] leading-none">
                {n}
              </span>
            ))}
          </span>
        </span>
      ))}
    </span>
  );
}
