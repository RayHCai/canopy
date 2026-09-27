/** POST /reviews/:id/uploads -- presigned PUT URLs the browser uploads photos/video to directly. */
import type { FastifyInstance } from "fastify";
import { prisma } from "../db.js";
import { NotFoundError } from "../errors.js";
import { uploadsRequestSchema } from "../schemas.js";
import { presignPut, uploadKey } from "../s3.js";

export async function uploadRoutes(app: FastifyInstance): Promise<void> {
  app.post<{ Params: { id: string } }>("/reviews/:id/uploads", async (req) => {
    const { id } = req.params;
    const body = uploadsRequestSchema.parse(req.body);

    const review = await prisma.review.findUnique({ where: { id } });
    if (!review) throw new NotFoundError(`review not found: ${id}`);

    const uploads = await Promise.all(
      body.files.map(async (file) => {
        const key = uploadKey(id, file.name);
        const url = await presignPut(key, file.contentType);
        return { name: file.name, key, url };
      }),
    );

    return { uploads };
  });
}
