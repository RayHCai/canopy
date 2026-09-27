import { AppShell } from "@/components/AppShell";
import Link from "next/link";

export default function NotFound() {
  return (
    <AppShell>
      <section className="pt-24">
        <p className="type-eyebrow anim-rise text-ink-3">Not found</p>
        <h1 className="type-display anim-rise mt-6 text-[clamp(46px,7vw,104px)]">
          No one lives here.
        </h1>
        <p className="anim-rise mt-8 max-w-[46ch] text-[19px] leading-snug text-ink-2">
          There is no member at this address. They may have been removed, or the link is wrong.
        </p>
        <Link
          href="/"
          transitionTypes={["nav-back"]}
          className="group anim-rise mt-12 inline-flex items-center gap-3 text-[17px] font-semibold"
        >
          <span aria-hidden className="transition-transform duration-500 ease-snap group-hover:-translate-x-1.5">
            ←
          </span>
          Back to the queue
        </Link>
      </section>
    </AppShell>
  );
}
