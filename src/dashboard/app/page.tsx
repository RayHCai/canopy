import { AppShell } from "@/components/AppShell";
import { QueueClient } from "@/components/queue/QueueClient";
import { ViewTransition } from "react";

const slide = {
  "nav-forward": "nav-forward",
  "nav-back": "nav-back",
  default: "none",
};

export default function QueuePage() {
  return (
    <AppShell>
      <ViewTransition enter={slide} exit={slide} default="none">
        <div>
          <QueueClient />
        </div>
      </ViewTransition>
    </AppShell>
  );
}
