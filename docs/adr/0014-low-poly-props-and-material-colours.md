# 14. Low-poly props authored in code; the viewer paints material colours

Date: 2026-09-26
Status: accepted

## Context

The model library's props (ADR 0007) were exported from elsewhere as dense,
voxel-like meshes -- a tree was 19,860 triangles of small cubes -- and the
generator had no way to rebuild them. Their colours never reached the screen
either: `AssetLibrary.build` splits a model by semantic class and paints every
face of a class object `CLASS_COLORS[cls]`, so a meter with twelve authored
materials drew as one flat yellow block. `spec.md` Module 1 describes exactly
that: "true colors for reveal + photos" are one colour per class, and the
reveal turns grey into the class colour.

That flatness is not an accident. ADR 0011 makes the detectors match on the
colour the ranger reports, and "colour is a strong cue because worldgen
colours are flat per class". Any richer colouring therefore has to leave what
the drones sense alone.

The ask was a "POLYGON"-style restyle of every asset except the drone: chunky
low-poly forms with real detail and several flat colours per prop.

## Decision

- **Props are authored by `scripts/build_props.py`.** Each prop is a union of
  convex parts (chamfered boxes, n-gon prisms, faceted domes, jittered
  icosphere lobes) built as convex hulls, which fixes outward winding for the
  viewer's back-face culling. Every builder asserts the bounding box the
  library was tuned to, so placement rules, `rules.yaml`'s battery box and the
  detectors' size bands see the same extents as before (within 6 mm). The
  authored house shell is one of these props now (`house_ranch.obj`), at the
  exact envelope of the home scenes' shell, so no seed's layout moves. A
  restyle is a diff to the script, and `--catalog` rebuilds the catalog sheet.
- **Material colours are carried as display data, not sensed data.** Each
  `SceneObject` gains `materials`, runs of consecutive faces with the authored
  MTL colour (or an `index.yaml` `palette:` entry for `parts:` models, or the
  class colour). `BuiltMesh` orders its faces by material and records the runs;
  the exported OBJ marks them with `usemtl`; the manifest round-trips them; the
  viewer's world payload expands them into a per-face `colors` array; and the
  page reveals each triangle into its own material colour. `SceneObject.color`
  is untouched, and it is still what the ranger reports, what photos shade with
  and what the reveal falls back to when `materials` is empty.
- **Nothing of class WALL stands more than 6 cm proud of the house's brick.**
  Wall-mounting rules read the WALL bounding box as the wall plane, so trims,
  shutters and frames are thin, and anything that must project further
  (gutters, downspouts, chimney, barge boards, the door canopy) is mapped to
  ROOF or DRIVEWAY by the house's per-model `materials` overrides.

## Consequences

- Surveyed-lot triangle counts fell from 145k-251k to 36k-144k a scene, with
  the tree at 1,000 triangles and the bush at 744. The reveal on vegetation now
  pops whole facets rather than wiping across a surface, which is the style.
- The viewer's grey-to-colour reveal ends in the art's colours, not the class
  palette. Photos in the SSR packet still use the class palette; giving them
  the material runs is a contained follow-up.
- Sensing is unchanged, so ADR 0011's detectors and their tuning hold. Shapes
  did change within the same boxes; the end-to-end detection test is the check
  that they hold in practice.
- The old home-scene shell is no longer referenced by the library; the
  `homes/` exports stay for `yard_authored` and as the record of the twelve
  authored service layouts ADR 0008 was calibrated against.
- The generated files under `assets/obj_export/assets/` are committed
  alongside the script, so a checkout works without running it; a change to
  the script is not complete until the files are regenerated and committed
  with it (`make props`).
