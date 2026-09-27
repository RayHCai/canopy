import type { ReactNode } from "react";

/**
 * One shape for "there is nothing to show, and here's why": an empty queue,
 * the API unreachable, a 500. Distinct from `NoReport` in MemberClient, which
 * means something narrower and more hopeful (this one home's survey
 * specifically is still in flight).
 */
export function StatusNotice({
  eyebrow,
  title,
  message,
  action,
}: {
  eyebrow: string;
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <section className="anim-rise border-t-2 border-ink pt-8">
      <p className="type-eyebrow text-ink-3">{eyebrow}</p>
      <p className="type-display mt-6 max-w-[18ch] text-[clamp(34px,4.6vw,64px)] text-ink-3">
        {title}
      </p>
      <p className="mt-6 max-w-[54ch] text-[18px] leading-relaxed text-ink-2">{message}</p>
      {action ? <div className="mt-10">{action}</div> : null}
    </section>
  );
}
