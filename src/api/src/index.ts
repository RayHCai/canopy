/**
 * Entry point: Fastify app wiring (ADR 0017). CORS is wide open -- the
 * viewer page and the dashboard each run on their own origin/port and
 * neither carries credentials this service needs to protect.
 */
import cors from "@fastify/cors";
import Fastify from "fastify";
import { ZodError } from "zod";
import { env } from "./env.js";
import { NotFoundError, BadRequestError } from "./errors.js";
import { healthRoutes } from "./routes/health.js";
import { mediaRoutes } from "./routes/media.js";
import { reviewRoutes } from "./routes/reviews.js";
import { uploadRoutes } from "./routes/uploads.js";

const app = Fastify({
  logger: true,
  // The run record carries per-drone sample tracks; a few MB of JSON is normal.
  bodyLimit: 20 * 1024 * 1024,
});

// @fastify/cors >= 10 defaults `methods` to GET,HEAD,POST, which silently
// fails the preflight for PUT /reviews/:id/run.
await app.register(cors, {
  origin: true,
  methods: ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
});

app.setErrorHandler((error, _req, reply) => {
  if (error instanceof ZodError) {
    reply.code(400).send({ error: error.issues.map((i) => `${i.path.join(".")}: ${i.message}`).join("; ") });
    return;
  }
  if (error instanceof NotFoundError) {
    reply.code(404).send({ error: error.message });
    return;
  }
  if (error instanceof BadRequestError) {
    reply.code(400).send({ error: error.message });
    return;
  }
  app.log.error(error);
  const message = error instanceof Error ? error.message : "internal error";
  reply.code(500).send({ error: message });
});

await app.register(healthRoutes);
await app.register(reviewRoutes);
await app.register(uploadRoutes);
await app.register(mediaRoutes);

await app.listen({ port: env.PORT, host: "0.0.0.0" });
