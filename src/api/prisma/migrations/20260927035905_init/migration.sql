-- CreateEnum
CREATE TYPE "ReviewStatus" AS ENUM ('AWAITING', 'READY', 'EMAIL_SENT');

-- CreateTable
CREATE TABLE "Review" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "address" TEXT NOT NULL,
    "lat" DOUBLE PRECISION,
    "lon" DOUBLE PRECISION,
    "siteId" TEXT,
    "status" "ReviewStatus" NOT NULL DEFAULT 'AWAITING',
    "seed" INTEGER NOT NULL,
    "droneCount" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "startedAt" TIMESTAMP(3),
    "locationMode" TEXT,
    "simDurationS" DOUBLE PRECISION,
    "mappedAtS" DOUBLE PRECISION,
    "coverageTotal" DOUBLE PRECISION,
    "coverageGround" DOUBLE PRECISION,
    "verdict" TEXT,
    "justification" TEXT,
    "videoKey" TEXT,
    "recommendation" JSONB,
    "sentEmail" JSONB,

    CONSTRAINT "Review_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "DroneTrack" (
    "id" TEXT NOT NULL,
    "reviewId" TEXT NOT NULL,
    "droneId" INTEGER NOT NULL,
    "samples" JSONB NOT NULL,
    "alive" BOOLEAN NOT NULL,
    "battery" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "DroneTrack_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "FleetEvent" (
    "id" TEXT NOT NULL,
    "reviewId" TEXT NOT NULL,
    "t" DOUBLE PRECISION NOT NULL,
    "droneId" INTEGER NOT NULL,
    "type" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "batteryPct" INTEGER NOT NULL,
    "task" TEXT NOT NULL,

    CONSTRAINT "FleetEvent_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Detection" (
    "id" TEXT NOT NULL,
    "reviewId" TEXT NOT NULL,
    "detId" INTEGER NOT NULL,
    "cls" TEXT NOT NULL,
    "color" JSONB NOT NULL,
    "center" JSONB NOT NULL,
    "size" JSONB NOT NULL,
    "yaw" DOUBLE PRECISION NOT NULL,
    "confidence" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "Detection_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SiteSuggestion" (
    "id" TEXT NOT NULL,
    "reviewId" TEXT NOT NULL,
    "rank" INTEGER NOT NULL,
    "pos" JSONB NOT NULL,
    "meter" JSONB NOT NULL,
    "yaw" DOUBLE PRECISION NOT NULL,
    "cost" DOUBLE PRECISION NOT NULL,
    "verdict" TEXT NOT NULL,
    "breakdown" JSONB NOT NULL,
    "warnings" JSONB NOT NULL,
    "photoKey" TEXT,
    "photoW" INTEGER,
    "photoH" INTEGER,
    "placementBoxes" JSONB,
    "summary" TEXT,

    CONSTRAINT "SiteSuggestion_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Blocker" (
    "id" TEXT NOT NULL,
    "siteId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "severity" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "requiredFix" TEXT NOT NULL,
    "box" JSONB NOT NULL,
    "trackId" INTEGER,

    CONSTRAINT "Blocker_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Review_createdAt_idx" ON "Review"("createdAt");

-- CreateIndex
CREATE INDEX "DroneTrack_reviewId_idx" ON "DroneTrack"("reviewId");

-- CreateIndex
CREATE INDEX "FleetEvent_reviewId_idx" ON "FleetEvent"("reviewId");

-- CreateIndex
CREATE INDEX "Detection_reviewId_idx" ON "Detection"("reviewId");

-- CreateIndex
CREATE INDEX "SiteSuggestion_reviewId_idx" ON "SiteSuggestion"("reviewId");

-- CreateIndex
CREATE UNIQUE INDEX "SiteSuggestion_reviewId_rank_key" ON "SiteSuggestion"("reviewId", "rank");

-- CreateIndex
CREATE INDEX "Blocker_siteId_idx" ON "Blocker"("siteId");

-- AddForeignKey
ALTER TABLE "DroneTrack" ADD CONSTRAINT "DroneTrack_reviewId_fkey" FOREIGN KEY ("reviewId") REFERENCES "Review"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "FleetEvent" ADD CONSTRAINT "FleetEvent_reviewId_fkey" FOREIGN KEY ("reviewId") REFERENCES "Review"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Detection" ADD CONSTRAINT "Detection_reviewId_fkey" FOREIGN KEY ("reviewId") REFERENCES "Review"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SiteSuggestion" ADD CONSTRAINT "SiteSuggestion_reviewId_fkey" FOREIGN KEY ("reviewId") REFERENCES "Review"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Blocker" ADD CONSTRAINT "Blocker_siteId_fkey" FOREIGN KEY ("siteId") REFERENCES "SiteSuggestion"("id") ON DELETE CASCADE ON UPDATE CASCADE;
