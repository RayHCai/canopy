"""Tests for canopy.worldgen.invariants.

Registration, the redraw-on-violation loop, and the review-only mission
checks, exercised through generate_field on lots tuned to be barely wide
enough, entirely too tight, or close but still workable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import Cls, Provenance, SiteBuilding, SiteSnapshot
from canopy.errors import SiteRejectedError
from canopy.worldgen import invariants
from canopy.worldgen import placement as rules
from canopy.worldgen.generate import generate_field
from canopy.worldgen.index_schema import load_library


def test_the_five_meter_invariants_are_registered() -> None:
    """The meter is the only role with hard rules today; all five are named here."""
    assert invariants.invariant_names() == [
        "meter_corner_clearance",
        "meter_height",
        "meter_on_exterior_wall",
        "meter_working_space",
        "one_meter",
    ]


def test_place_calls_a_rule_with_no_invariants_exactly_once(
    monkeypatch: pytest.MonkeyPatch, cfg: Config
) -> None:
    """A role whose rule has no registered invariant is simply called once, untouched."""
    library = load_library()
    role = next(r for r in library.roles if r.rule != "service_assembly")
    spec = library.choose(role.tag, np.random.default_rng(0))
    calls: list[int] = []

    def counting_rule(ctx: rules.PlacementContext) -> list[rules.Placement]:
        del ctx
        calls.append(1)
        return []

    monkeypatch.setattr(invariants, "get_rule", lambda _name: counting_rule)

    ctx = rules.PlacementContext(
        rng=np.random.default_rng(0),
        cfg=cfg,
        library=library,
        spec=spec,
        role=role,
        n=0,
        lot_bounds=np.zeros((2, 3), dtype=np.float64),
    )
    assert invariants.place(ctx) == []
    assert len(calls) == 1


def _bbox_centre_x(path: Path) -> float:
    """Midpoint of an OBJ file's vertices along world x."""
    xs = [
        float(line.split()[1])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    return (min(xs) + max(xs)) / 2.0


# ---------------------------------------------------------------------------
# A lot that is barely wide enough
# ---------------------------------------------------------------------------
#: Half this lot's width is 6.6 m against the fixture house's 6 m half-width:
#: only 0.6 m stands between the main block's outer left/right wall and the
#: lot line, well under worldgen.invariants.meter_working_space_m (0.9 m). The
#: notch between the main block and the wing, and every front/back wall, has
#: room to spare, and the meter's odds skip the walls that do not.
_NARROW_X_LOT_M = (13.2, 36.0)
_NARROW_X_SEEDS = range(8)


@pytest.mark.slow
def test_a_barely_wide_lot_never_ends_with_the_meter_on_the_tight_outer_wall(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    snapshot = make_site(lot_m=_NARROW_X_LOT_M)
    for seed in _NARROW_X_SEEDS:
        manifest = generate_field(seed, cfg, out_dir=tmp_path / str(seed), site=snapshot)
        meter = next(o for o in manifest.objects if o.obj_id == manifest.gt_meter_id)
        assert meter.wall_normal is not None
        half_x = max(abs(x) for x, _ in manifest.footprint)
        meter_x = _bbox_centre_x(Path(meter.mesh_path))
        on_tight_outer_wall = (
            abs(abs(meter_x) - half_x) < 0.3 and abs(float(meter.wall_normal[0])) > 0.5
        )
        assert not on_tight_outer_wall, (
            f"seed {seed}: meter at x={meter_x:.2f} is on the tight wall"
        )


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
#: How far the forced bad draw lifts the meter: its centre then sits near
#: 2.6 m, well above worldgen.invariants.meter_centre_height_m.
_LIFT_M = 1.0


def test_a_draw_that_breaks_an_invariant_is_redrawn_and_marked_repaired(
    cfg: Config,
    site_snapshot: SiteSnapshot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The first service draw hangs the meter too high; the redraw must fix and mark it.

    Forcing the bad draw, rather than hunting for a seed that happens to
    produce one, keeps this about the repair loop itself: the rules' own
    priors are written to make violations rare.
    """
    real_get_rule = rules.get_rule
    draws: list[int] = []

    def lifting_get_rule(name: str) -> rules.PlacementRule:
        rule = real_get_rule(name)
        if name != "service_assembly":
            return rule

        def first_draw_too_high(ctx: rules.PlacementContext) -> list[rules.Placement]:
            found = rule(ctx)
            draws.append(1)
            if len(draws) > 1:
                return found
            lift = np.array([0.0, 0.0, _LIFT_M])
            return [
                dataclasses.replace(p, pos=p.pos + lift) if p.spec.cls is Cls.METER else p
                for p in found
            ]

        return first_draw_too_high

    monkeypatch.setattr(invariants, "get_rule", lifting_get_rule)
    manifest = generate_field(1, cfg, out_dir=tmp_path, site=site_snapshot)

    meter = next(o for o in manifest.objects if o.obj_id == manifest.gt_meter_id)
    assert meter.provenance is Provenance.REPAIRED
    zs = [
        float(line.split()[3])
        for line in Path(meter.mesh_path).read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ]
    lo, hi = cfg.worldgen.invariants.meter_centre_height_m
    assert lo <= (min(zs) + max(zs)) / 2.0 <= hi
    assert len(draws) >= 2


# ---------------------------------------------------------------------------
# No wall has room, anywhere
# ---------------------------------------------------------------------------
#: A plain 12 x 14 m box on a lot 0.3 m larger all round: every wall faces a
#: lot line far closer than worldgen.invariants.meter_working_space_m, and no
#: notch offers a way out, so no seed and no redraw can fit the meter.
#: Generation must refuse rather than place a meter nobody can stand before.
_NO_ROOM_LOT_M = (12.6, 14.6)
_NO_ROOM_SEED = 17


def test_a_lot_with_no_room_anywhere_is_rejected(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    box = SiteBuilding(
        footprint=np.array([[-6.0, -7.0], [6.0, -7.0], [6.0, 7.0], [-6.0, 7.0]]),
        levels=2,
        height_m=None,
        roof_shape="gabled",
        source="fixture:way/1",
    )
    snapshot = make_site(house=box, lot_m=_NO_ROOM_LOT_M)
    with pytest.raises(SiteRejectedError, match="meter_working_space"):
        generate_field(_NO_ROOM_SEED, cfg, out_dir=tmp_path, site=snapshot)


# ---------------------------------------------------------------------------
# review(): reported, never repaired
# ---------------------------------------------------------------------------
#: Half this lot's width is 7.3 m against the house's 6 m half-width: 1.3 m
#: clear of the outer wall, enough for the 0.9 m hard rule but, once the 0.5 m
#: geofence inset is taken off, short of the 2.1 m INSPECT's close-up wants.
_CLOSE_BUT_WORKABLE_LOT_M = (14.6, 36.0)
_CLOSE_BUT_WORKABLE_SEED = 0


def test_review_notes_a_close_but_still_workable_meter_approach(
    cfg: Config, make_site: Callable[..., SiteSnapshot], tmp_path: Path
) -> None:
    snapshot = make_site(lot_m=_CLOSE_BUT_WORKABLE_LOT_M)
    manifest = generate_field(_CLOSE_BUT_WORKABLE_SEED, cfg, out_dir=tmp_path, site=snapshot)
    assert len(manifest.notes) == 1
    assert "meter approach" in manifest.notes[0]


def test_review_reports_nothing_when_the_meter_faces_open_ground(
    cfg: Config, site_snapshot: SiteSnapshot, tmp_path: Path
) -> None:
    """The default fixture lot is roomy on every side, so no seed needs a note."""
    manifest = generate_field(1, cfg, out_dir=tmp_path, site=site_snapshot)
    assert manifest.notes == ()
