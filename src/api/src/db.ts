/** Single shared Prisma client, per Prisma's recommended pattern for a long-lived server process. */
import { PrismaClient } from "@prisma/client";

export const prisma = new PrismaClient();
