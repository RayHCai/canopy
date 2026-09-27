import type { MemberEmailDraft } from "./types";

const MOCK_SEND_DELAY_MS = 1000;

/**
 * TODO: swap this mock for a real send, e.g.:
 *   const resend = new Resend(process.env.RESEND_API_KEY);
 *   await resend.emails.send({ from: "Base Power <noreply@basepower.com>", to: draft.to, subject: draft.subject, html: draft.html });
 * The delay stands in for network latency so the UI's sending state is exercised.
 */
export async function sendEmail(draft: MemberEmailDraft): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, MOCK_SEND_DELAY_MS));
  console.log("[Canopy] sendEmail mock", {
    to: draft.to,
    subject: draft.subject,
    body: draft.body,
    html: draft.html,
    placementPhotoId: draft.placementPhoto.id,
    blockerPhotoIds: draft.blockerPhotos.map((p) => p.id),
  });
}
