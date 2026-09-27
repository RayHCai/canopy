"use client";

import { CheckGlyph, CrossGlyph, WarnGlyph } from "@/components/ui/glyphs";
import { SurveyPhoto, Viewfinder } from "@/components/ui/SurveyPhoto";
import { splitOptionLabel } from "@/lib/format";
import type { PlacementPhoto } from "@/lib/types";

function RadioMark({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden
      className={`mt-0.5 flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full border-2 transition-colors duration-300 ${
        checked ? "border-signal" : "border-ink-3 group-hover:border-ink"
      }`}
    >
      <span
        className={`h-2.5 w-2.5 rounded-full bg-signal transition-transform duration-500 ease-spring ${
          checked ? "scale-100" : "scale-0"
        }`}
      />
    </span>
  );
}

export function PlacementChooser({
  name,
  photos,
  selectedId,
  locked,
  recommendation,
  onSelect,
}: {
  name: string;
  photos: PlacementPhoto[];
  selectedId: string | null;
  locked: boolean;
  recommendation: { placementPhotoId: string; reason: string } | null;
  onSelect: (photo: PlacementPhoto, photoEl: HTMLElement | null) => void;
}) {
  return (
    <fieldset disabled={locked} className="min-w-0">
      <legend className="sr-only">Battery and disconnect placement</legend>
      {recommendation ? (
        <p className="-mt-4 mb-10 max-w-[68ch] border-l-2 border-signal pl-4 text-[16px] leading-relaxed text-ink-2">
          <span className="type-eyebrow mb-1.5 block text-signal">Swarm recommends</span>
          {recommendation.reason}
        </p>
      ) : null}
      <div
        className={`grid gap-x-10 gap-y-14 ${
          photos.length > 1 ? "sm:grid-cols-2" : "max-w-[620px]"
        }`}
      >
        {photos.map((photo, i) => {
          const checked = photo.id === selectedId;
          const dimmed = selectedId !== null && !checked;
          const { option, place } = splitOptionLabel(photo.label);

          return (
            <label
              key={photo.id}
              data-locked={locked || undefined}
              className={`vf-aim group block ${locked ? "cursor-default" : "cursor-pointer"}`}
            >
              <input
                type="radio"
                name={name}
                value={photo.id}
                checked={checked}
                onChange={(e) =>
                  onSelect(
                    photo,
                    e.currentTarget.parentElement?.querySelector<HTMLElement>("[data-photo]") ??
                      null,
                  )
                }
                className="sr-only"
              />
              <Viewfinder active={checked}>
                <div
                  data-photo
                  className={`transition-[filter,opacity] duration-500 ease-snap ${
                    dimmed ? "opacity-45 grayscale" : ""
                  }`}
                >
                  <SurveyPhoto
                    src={photo.imageUrl}
                    alt={`${photo.label}, with the proposed battery and disconnect footprints`}
                    proposals={photo.placementBoxes}
                    eager={i < 2}
                  >
                    {photo.recommended ? (
                      <span className="type-eyebrow absolute left-3 top-3 inline-flex items-center gap-1.5 bg-signal px-2 py-1 text-white">
                        <CheckGlyph className="h-3 w-3" strokeWidth={2.4} />
                        Recommended
                      </span>
                    ) : null}
                    {photo.verdict === "reject" ? (
                      <span className="type-eyebrow absolute right-3 top-3 inline-flex items-center gap-1.5 bg-critical px-2 py-1 text-white">
                        <CrossGlyph className="h-3 w-3" strokeWidth={2.4} />
                        Fails clearance
                      </span>
                    ) : null}
                  </SurveyPhoto>
                </div>
              </Viewfinder>

              <div className="mt-6 flex items-start gap-4">
                <RadioMark checked={checked} />
                <div className="min-w-0">
                  <p className="text-[20px] font-semibold leading-tight tracking-[-0.02em]">
                    {option}
                    {place ? <span className="font-normal text-ink-2"> · {place}</span> : null}
                  </p>
                  <dl className="mt-2.5 flex flex-wrap gap-x-6 gap-y-1 font-mono text-[13px]">
                    <div>
                      <dt className="sr-only">Distance to meter</dt>
                      <dd>
                        <span className="font-medium text-ink">{photo.distanceToMeterFt} ft</span>{" "}
                        <span className="text-ink-3">to meter</span>
                      </dd>
                    </div>
                    <div>
                      <dt className="sr-only">Clearance</dt>
                      <dd className="flex items-center gap-1.5">
                        {/* A manual-review site has not failed clearance; it needs a person to confirm it. */}
                        {photo.verdict === "pass" ? (
                          <>
                            <CheckGlyph className="h-3.5 w-3.5 text-good" strokeWidth={2.2} />
                            <span className="text-good-ink">Clearance passes</span>
                          </>
                        ) : photo.verdict === "manual_review" ? (
                          <>
                            <WarnGlyph className="h-3.5 w-3.5 text-warning" strokeWidth={2.2} />
                            <span className="text-warning-ink">Clearance needs review</span>
                          </>
                        ) : (
                          <>
                            <CrossGlyph className="h-3.5 w-3.5 text-critical" strokeWidth={2.2} />
                            <span className="text-critical">Clearance fails</span>
                          </>
                        )}
                      </dd>
                    </div>
                  </dl>
                  {photo.summary ? (
                    <p className="mt-3 max-w-[52ch] text-[15px] leading-relaxed text-ink-2">
                      {photo.summary}
                    </p>
                  ) : null}
                </div>
              </div>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
