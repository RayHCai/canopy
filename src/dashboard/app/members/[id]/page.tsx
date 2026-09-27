import { AppShell } from "@/components/AppShell";
import { StatusNotice } from "@/components/StatusNotice";
import { MemberClient } from "@/components/member/MemberClient";
import { getReview } from "@/lib/api";
import { notFound } from "next/navigation";
import { ViewTransition } from "react";

const slide = {
  "nav-forward": "nav-forward",
  "nav-back": "nav-back",
  default: "none",
};

export default async function MemberPage({
  params,
}: PageProps<"/members/[id]">) {
  const { id } = await params;

  // `getReview` resolves null only for a 404; a network failure (API down,
  // wrong host) throws instead, which gets its own message rather than the
  // Next.js error boundary's generic crash screen.
  let member;
  try {
    member = await getReview(id);
  } catch (err) {
    return (
      <AppShell>
        <div className="pt-12 sm:pt-16">
          <StatusNotice
            eyebrow="Review API offline"
            title="Can't reach the review API"
            message={err instanceof Error ? err.message : "Something went wrong loading this member."}
          />
        </div>
      </AppShell>
    );
  }

  if (!member) {
    notFound();
  }

  return (
    <AppShell>
      <ViewTransition key={member.id} enter={slide} exit={slide} default="none">
        <div>
          <MemberClient initialMember={member} />
        </div>
      </ViewTransition>
    </AppShell>
  );
}
