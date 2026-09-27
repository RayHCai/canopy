import type { BlockerPhoto, Member, PlacementPhoto } from "./types";

/** Member data is interpolated into markup, so it must not be able to become markup. */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function photoBlock(imageUrl: string, alt: string): string {
  return `<img src="${escapeHtml(imageUrl)}" alt="${escapeHtml(alt)}" width="552" style="display:block;width:100%;max-width:552px;height:auto;border-radius:8px;margin:16px 0;" />`;
}

/**
 * Builds the email body as a self-contained HTML fragment with inline styles,
 * so it renders consistently once dropped into a real ESP (e.g. Resend).
 */
export function buildEmailHtml({
  member,
  placementPhoto,
  blockerPhotos,
  fixes,
}: {
  member: Member;
  placementPhoto: PlacementPhoto;
  blockerPhotos: BlockerPhoto[];
  fixes: string[];
}): string {
  const firstName = escapeHtml(member.name.split(" ")[0]);
  const placementLabel = escapeHtml(placementPhoto.label);
  const hasBlockers = fixes.length > 0;

  const intro = hasBlockers
    ? `<p style="margin:0 0 16px;font-size:16px;line-height:1.5;color:#27272a;">Thank you for hosting our drone site survey. We reviewed your equipment wall and recommend the battery location shown below (${placementLabel}).</p>`
    : `<p style="margin:0 0 16px;font-size:16px;line-height:1.5;color:#27272a;">We reviewed your drone survey photos and recommend the battery location shown below (${placementLabel}).</p>`;

  const blockerSection = hasBlockers
    ? `
      <p style="margin:24px 0 12px;font-size:16px;line-height:1.5;color:#27272a;">We need your help addressing the following before install:</p>
      ${blockerPhotos.map((photo) => photoBlock(photo.imageUrl, "Site blocker photo")).join("")}
      <ol style="margin:0 0 16px;padding-left:20px;font-size:16px;line-height:1.7;color:#27272a;">
        ${fixes.map((fix) => `<li>${escapeHtml(fix)}</li>`).join("")}
      </ol>
      <p style="margin:0 0 16px;font-size:16px;line-height:1.5;color:#27272a;">Once these are done, reply with a photo and we'll move you to the next step.</p>
    `
    : `<p style="margin:0 0 16px;font-size:16px;line-height:1.5;color:#27272a;">Congratulations, your home is approved for your Base Power battery. Please wait for next steps from our team.</p>`;

  return `
    <div style="max-width:600px;margin:0 auto;padding:32px 24px;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;">
      <div style="font-size:20px;font-weight:700;letter-spacing:-0.01em;color:#0f766e;">Base Power</div>
      <div style="height:3px;background:#0f766e;margin:16px 0 24px;border-radius:2px;"></div>
      <p style="margin:0 0 16px;font-size:16px;line-height:1.5;color:#18181b;">Hi ${firstName},</p>
      ${intro}
      ${photoBlock(placementPhoto.imageUrl, "Recommended battery placement")}
      ${blockerSection}
      <div style="margin-top:32px;padding-top:16px;border-top:1px solid #e4e4e7;">
        <p style="margin:0;font-size:13px;line-height:1.5;color:#a1a1aa;">Base Power Site Survey Review</p>
        <p style="margin:4px 0 0;font-size:13px;line-height:1.5;color:#a1a1aa;">This is an automated message about your battery install.</p>
      </div>
    </div>
  `.trim();
}
