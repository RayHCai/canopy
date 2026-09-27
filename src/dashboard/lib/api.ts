import type { Member } from "./types";

/**
 * ADR 0017: the dashboard reads only the review API. Base URL is public
 * (browser and server both hit it directly; there is no dashboard backend
 * of its own to proxy through).
 */
const API_BASE =
  process.env.NEXT_PUBLIC_CANOPY_API_URL?.replace(/\/+$/, "") ??
  "http://localhost:4000";

/**
 * Thrown for every non-2xx response and for the fetch itself failing (the
 * API process isn't up, wrong host, etc). `status` is absent for the latter
 * case, which callers use to tell "API said no" from "API isn't there."
 */
export class ApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: "no-store",
      headers: init?.body
        ? { "Content-Type": "application/json", ...init.headers }
        : init?.headers,
    });
  } catch {
    throw new ApiError(
      `Could not reach the Canopy review API at ${API_BASE}. Is it running?`,
    );
  }

  if (!res.ok) {
    let message = `${path} failed with ${res.status}`;
    try {
      const body: unknown = await res.json();
      if (body && typeof body === "object" && "error" in body) {
        const { error } = body as { error: unknown };
        if (typeof error === "string" && error) message = error;
      }
    } catch {
      // Not a JSON error body; keep the generic message.
    }
    throw new ApiError(message, res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** `GET /reviews`, newest first. */
export function listReviews(): Promise<Member[]> {
  return request<Member[]>("/reviews");
}

/** `GET /reviews/:id`. Resolves `null` for a 404 rather than throwing, so callers can `notFound()`. */
export async function getReview(id: string): Promise<Member | null> {
  try {
    return await request<Member>(`/reviews/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export interface RecordEmailSentInput {
  subject: string;
  approved: boolean;
  placementPhotoId: string;
  blockerPhotoIds: string[];
}

/** `POST /reviews/:id/email`, returning the member with `sent` and `reportStatus` updated. */
export function recordEmailSent(
  id: string,
  body: RecordEmailSentInput,
): Promise<Member> {
  return request<Member>(`/reviews/${encodeURIComponent(id)}/email`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
