import { AppShell } from "@/components/AppShell";
import { MemberClient } from "@/components/member/MemberClient";
import { getMemberById } from "@/lib/mockData";
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
  const member = getMemberById(id);

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
