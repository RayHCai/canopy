/** GET /media/*key -- 302 to a presigned GET, so the browser fetches straight from S3. */
import type { FastifyInstance } from "fastify";
import { presignGet } from "../s3.js";

export async function mediaRoutes(app: FastifyInstance): Promise<void> {
  app.get<{ Params: { "*": string } }>("/media/*", async (req, reply) => {
    const key = req.params["*"];
    const url = await presignGet(key);
    return reply.redirect(url, 302);
  });
}
