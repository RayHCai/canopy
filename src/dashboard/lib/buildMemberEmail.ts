import { buildEmailHtml } from "./emailTemplate";
import type {
  BlockerPhoto,
  Member,
  MemberEmailDraft,
  PlacementPhoto,
} from "./types";

export const SUBJECT_ACTION_NEEDED =
  "Action needed for your Base Power battery install";
export const SUBJECT_APPROVED = "Congratulations, you're approved!";

/** The email asks for fixes only when an included photo actually has blockers. */
export function includesBlockers(blockerPhotos: BlockerPhoto[]): boolean {
  return blockerPhotos.some((p) => p.blockers.length > 0);
}

export function collectFixes(blockerPhotos: BlockerPhoto[]): string[] {
  const fixes: string[] = [];
  for (const photo of blockerPhotos) {
    for (const blocker of photo.blockers) {
      fixes.push(blocker.requiredFix);
    }
  }
  return [...new Set(fixes)];
}

export function buildMemberEmail(
  member: Member,
  placementPhoto: PlacementPhoto,
  selectedBlockerPhotos: BlockerPhoto[],
): MemberEmailDraft {
  const firstName = member.name.split(" ")[0];
  const hasBlockers = includesBlockers(selectedBlockerPhotos);
  const fixes = collectFixes(selectedBlockerPhotos);

  const html = buildEmailHtml({
    member,
    placementPhoto,
    blockerPhotos: selectedBlockerPhotos,
    fixes,
  });

  if (hasBlockers) {
    const fixList = fixes.map((fix, i) => `${i + 1}. ${fix}`).join("\n");
    return {
      to: member.email,
      subject: SUBJECT_ACTION_NEEDED,
      body:
        `Hi ${firstName},\n\n` +
        `Thank you for hosting our drone site survey. We reviewed your equipment wall and recommend the battery location shown in the attached placement photo (${placementPhoto.label}).\n\n` +
        `We need your help addressing the following items shown in the attached blocker photos:\n\n` +
        `${fixList}\n\n` +
        `Once these are done, reply with a photo and we'll move you to the next step.\n\n` +
        `Base Power Site Survey Review`,
      html,
      placementPhoto,
      blockerPhotos: selectedBlockerPhotos,
    };
  }

  return {
    to: member.email,
    subject: SUBJECT_APPROVED,
    body:
      `Hi ${firstName},\n\n` +
      `We reviewed your drone survey photos and recommend the battery location shown in the attached placement photo (${placementPhoto.label}).\n\n` +
      `Congratulations, your home is approved for your Base Power battery. Please wait for next steps from our team.\n\n` +
      `Base Power Site Survey Review`,
    html,
    placementPhoto,
    blockerPhotos: selectedBlockerPhotos,
  };
}
