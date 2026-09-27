"""Whether an address is a home: the evidence, the vocabulary, and the score.

The address text itself says almost nothing -- "12 Oak Street" scans the same
whether it is a house or a dentist's office -- so the evidence this reads is
about what stands *at* the address: how the building and its surroundings are
mapped. :func:`score_residential` folds that evidence into log-odds
(:class:`~canopy.config.ResidentialCfg` supplies the weights) and reports a
three-valued call, because thin evidence should read as "ask", never as a
confident wrong answer in either direction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from canopy.config import ResidentialCfg
from canopy.contracts import ResidentialDecision, ResidentialVerdict

__all__ = ["ResidentialFeatures", "mentions_suite", "score_residential"]

#: OSM ``building`` values that are a single house.
_HOUSE_TAGS = frozenset(
    {"house", "detached", "semidetached_house", "terrace", "bungalow", "cabin", "farm"}
)
#: Residential, but not a single house (a duplex or a block of flats).
_RESIDENTIAL_NOT_HOUSE_TAGS = frozenset({"residential", "apartments", "dormitory"})
#: Plainly not residential.
_NONRESIDENTIAL_TAGS = frozenset(
    {
        "commercial",
        "retail",
        "industrial",
        "office",
        "warehouse",
        "supermarket",
        "kiosk",
        "school",
        "university",
        "college",
        "kindergarten",
        "hospital",
        "church",
        "chapel",
        "cathedral",
        "mosque",
        "synagogue",
        "temple",
        "religious",
        "public",
        "civic",
        "government",
        "hotel",
        "train_station",
        "transportation",
        "fire_station",
        "parking",
        "service",
        "manufacture",
        "factory",
        "sports_hall",
        "stadium",
    }
)
#: ``landuse`` values, likewise.
_LANDUSE_RESIDENTIAL = frozenset({"residential"})
_LANDUSE_NONRESIDENTIAL = frozenset({"commercial", "retail", "industrial"})

#: A unit or suite number in free text, e.g. "Suite 200" or "Ste 4B".
_SUITE_RE = re.compile(r"\bsuite\b|\bste\b", re.IGNORECASE)
#: At or below this many storeys, a building reads as low-rise.
_LOW_RISE_MAX_LEVELS = 3


@dataclass(frozen=True, slots=True)
class ResidentialFeatures:
    """What the map data says at and around an address, for :func:`score_residential`."""

    building_tag: str | None
    """The target building's own ``building=*`` value; ``None`` if untagged."""
    building_has_poi_tags: bool
    """Whether the building itself also carries a ``shop``, ``office`` or ``amenity`` key."""
    pois_inside: int
    """Shop, office or amenity nodes mapped inside the building's footprint."""
    landuse_here: tuple[str, ...]
    """``landuse`` values of any area containing the address pin."""
    area_m2: float | None
    """Footprint area, for :attr:`~canopy.config.ResidentialCfg.house_area_m2`."""
    levels: int | None
    """Storeys above ground, as mapped or inferred; ``None`` if unknown."""
    suite: bool
    """Whether the address (or the search text that found it) names a suite."""


def mentions_suite(text: str) -> bool:
    """Whether ``text`` names a unit or suite number, which homes rarely have."""
    return _SUITE_RE.search(text) is not None


def _logistic(x: float) -> float:
    """Map log-odds ``x`` to a probability in ``(0, 1)`` (the standard logistic function)."""
    return 1.0 / (1.0 + math.exp(-x))


def _building_reason(tag: str | None, cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for the target building's own ``building=*`` tag, if it says anything."""
    if tag in _HOUSE_TAGS:
        return cfg.w_building_house, f"the building is mapped as {tag!r}, a house type"
    if tag in _RESIDENTIAL_NOT_HOUSE_TAGS:
        return (
            cfg.w_building_residential,
            f"the building is mapped as {tag!r}, residential but not a single house",
        )
    if tag in _NONRESIDENTIAL_TAGS:
        return cfg.w_building_nonresidential, f"the building is mapped as {tag!r}, not residential"
    return None


def _poi_reason(features: ResidentialFeatures, cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for a point of interest on or inside the building, if there is one."""
    if features.pois_inside > 0:
        return (
            cfg.w_poi_inside,
            f"{features.pois_inside} shop, office or amenity point(s) are mapped "
            "inside the footprint",
        )
    if features.building_has_poi_tags:
        return cfg.w_poi_inside, "the building itself carries a shop, office or amenity tag"
    return None


def _landuse_reason(landuse_here: tuple[str, ...], cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for the ``landuse`` at the pin, if it says anything either way."""
    if any(v in _LANDUSE_RESIDENTIAL for v in landuse_here):
        return cfg.w_landuse_residential, "the address lies in a landuse=residential area"
    if any(v in _LANDUSE_NONRESIDENTIAL for v in landuse_here):
        return cfg.w_landuse_nonresidential, "the address lies in a non-residential landuse area"
    return None


def _area_reason(area_m2: float | None, cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for a house-sized footprint, if the area is known and fits."""
    lo, hi = cfg.house_area_m2
    if area_m2 is not None and lo <= area_m2 <= hi:
        return cfg.w_house_sized, f"the footprint is {area_m2:.0f} m^2, a house-sized building"
    return None


def _levels_reason(levels: int | None, cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for a low-rise building, if the storey count is known and low."""
    if levels is not None and levels <= _LOW_RISE_MAX_LEVELS:
        return cfg.w_low_rise, f"the building has {levels} storey(s), a low-rise building"
    return None


def _suite_reason(suite: bool, cfg: ResidentialCfg) -> tuple[float, str] | None:
    """Log-odds and reason for the address naming a suite, if it does."""
    if suite:
        return cfg.w_unit_suite, "the address names a suite or unit, which homes rarely have"
    return None


def score_residential(features: ResidentialFeatures, cfg: ResidentialCfg) -> ResidentialVerdict:
    """Score how likely an address is a home, from what is mapped there.

    Every applicable piece of evidence contributes its configured log-odds
    weight (see :class:`~canopy.config.ResidentialCfg`); the sum runs through
    a logistic to give a probability, which two thresholds turn into a
    three-valued decision.

    Parameters
    ----------
    features
        What the fetch step found at and around the address.
    cfg
        Weights and decision thresholds.

    Returns
    -------
    ResidentialVerdict
        The probability, the decision, and the reasons, strongest first.
    """
    reasons = [
        reason
        for reason in (
            _building_reason(features.building_tag, cfg),
            _poi_reason(features, cfg),
            _landuse_reason(features.landuse_here, cfg),
            _area_reason(features.area_m2, cfg),
            _levels_reason(features.levels, cfg),
            _suite_reason(features.suite, cfg),
        )
        if reason is not None
    ]

    logit = sum(weight for weight, _ in reasons)
    p = _logistic(logit)
    if p >= cfg.accept_p:
        decision = ResidentialDecision.ACCEPT
    elif p <= cfg.reject_p:
        decision = ResidentialDecision.REJECT
    else:
        decision = ResidentialDecision.ASK

    # Most influential first, ties broken by the order evaluated above.
    ordered = sorted(reasons, key=lambda item: -abs(item[0]))
    sentences = tuple(f"{text} ({weight:+.2f} log-odds)." for weight, text in ordered)
    if not sentences:
        sentences = ("no mapped evidence bears on whether this is a home, either way.",)
    return ResidentialVerdict(p_residential=p, decision=decision, reasons=sentences)
