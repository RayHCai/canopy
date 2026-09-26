/**
 * Where the team picks a member up outside Canopy. Set these in .env.local:
 *   NEXT_PUBLIC_HUBSPOT_PORTAL_ID             the number after /contacts/ in any HubSpot URL
 *   NEXT_PUBLIC_SLACK_SITE_SURVEY_REVIEW_URL  "Copy link" on the site survey review channel
 * Until they're set, the links render disabled instead of pointing somewhere wrong.
 */
const hubspotPortalId = process.env.NEXT_PUBLIC_HUBSPOT_PORTAL_ID || null;

export const siteSurveyReviewSlackUrl =
  process.env.NEXT_PUBLIC_SLACK_SITE_SURVEY_REVIEW_URL || null;

/**
 * Opens HubSpot's contact search for the member's email. Searching by email
 * (rather than storing a contact ID) means the link can only land on this person.
 */
export function hubspotContactSearchUrl(email: string): string | null {
  if (!hubspotPortalId) return null;
  return `https://app.hubspot.com/contacts/${encodeURIComponent(
    hubspotPortalId,
  )}/objects/0-1/views/all/list?query=${encodeURIComponent(email)}`;
}
