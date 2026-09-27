/**
 * S3 client and presign helpers.
 *
 * The viewer page PUTs photo/video blobs directly to S3 via presigned URLs
 * (ADR 0017: "a browser can PUT a blob to a presigned URL without Python
 * growing an HTTP client"), and the dashboard fetches them through
 * `GET /media/*`, which redirects to a presigned GET. With `S3_ENDPOINT`
 * unset, this talks to real AWS S3; set it to point at local MinIO.
 */
import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { env } from "./env.js";

export const s3 = new S3Client({
  region: env.S3_REGION,
  endpoint: env.S3_ENDPOINT,
  forcePathStyle: env.S3_FORCE_PATH_STYLE,
  credentials: {
    accessKeyId: env.S3_ACCESS_KEY_ID,
    secretAccessKey: env.S3_SECRET_ACCESS_KEY,
  },
});

const FIFTEEN_MINUTES_S = 15 * 60;
const ONE_HOUR_S = 60 * 60;

/** Build the review-scoped object key for an uploaded file. */
export function uploadKey(reviewId: string, name: string): string {
  return `reviews/${reviewId}/${name}`;
}

/** Presigned PUT URL a browser can upload a blob to directly, valid 15 minutes. */
export async function presignPut(key: string, contentType: string): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: env.S3_BUCKET,
    Key: key,
    ContentType: contentType,
  });
  return getSignedUrl(s3, command, { expiresIn: FIFTEEN_MINUTES_S });
}

/** Presigned GET URL, valid 1 hour -- what `GET /media/*` redirects to. */
export async function presignGet(key: string): Promise<string> {
  const command = new GetObjectCommand({ Bucket: env.S3_BUCKET, Key: key });
  return getSignedUrl(s3, command, { expiresIn: ONE_HOUR_S });
}
