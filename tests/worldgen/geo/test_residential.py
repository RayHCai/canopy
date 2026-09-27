"""score_residential: the evidence table and all three decisions."""

from __future__ import annotations

import math

import pytest

from canopy.config import ResidentialCfg
from canopy.contracts import ResidentialDecision
from canopy.worldgen.geo.residential import ResidentialFeatures, mentions_suite, score_residential

#: A synthetic, easy-to-hand-check set of weights, independent of the shipped
#: config: this test is about the scoring algorithm, not the tuned defaults.
_CFG = ResidentialCfg(
    accept_p=0.8,
    reject_p=0.2,
    house_area_m2=(40.0, 400.0),
    w_building_house=2.0,
    w_building_residential=1.0,
    w_building_nonresidential=-3.0,
    w_poi_inside=-2.0,
    w_landuse_residential=1.0,
    w_landuse_nonresidential=-2.0,
    w_house_sized=0.5,
    w_low_rise=0.5,
    w_unit_suite=-1.0,
)

_NO_EVIDENCE = ResidentialFeatures(
    building_tag=None,
    building_has_poi_tags=False,
    pois_inside=0,
    landuse_here=(),
    area_m2=None,
    levels=None,
    suite=False,
)


def _logit_to_p(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


def test_no_evidence_asks_and_says_so() -> None:
    verdict = score_residential(_NO_EVIDENCE, _CFG)
    assert verdict.p_residential == pytest.approx(0.5)
    assert verdict.decision is ResidentialDecision.ASK
    assert len(verdict.reasons) == 1
    assert "no mapped evidence" in verdict.reasons[0]


def test_a_house_tagged_building_with_house_sized_area_and_low_rise_accepts() -> None:
    features = ResidentialFeatures(
        building_tag="house",
        building_has_poi_tags=False,
        pois_inside=0,
        landuse_here=("residential",),
        area_m2=150.0,
        levels=2,
        suite=False,
    )
    verdict = score_residential(features, _CFG)
    expected_logit = (
        _CFG.w_building_house + _CFG.w_landuse_residential + _CFG.w_house_sized + _CFG.w_low_rise
    )
    assert verdict.p_residential == pytest.approx(_logit_to_p(expected_logit))
    assert verdict.decision is ResidentialDecision.ACCEPT
    # Strongest evidence first.
    assert "house" in verdict.reasons[0]


def test_a_shop_building_with_a_poi_inside_and_commercial_landuse_rejects() -> None:
    features = ResidentialFeatures(
        building_tag="commercial",
        building_has_poi_tags=True,
        pois_inside=2,
        landuse_here=("commercial",),
        area_m2=None,
        levels=None,
        suite=False,
    )
    verdict = score_residential(features, _CFG)
    expected_logit = (
        _CFG.w_building_nonresidential + _CFG.w_poi_inside + _CFG.w_landuse_nonresidential
    )
    assert verdict.p_residential == pytest.approx(_logit_to_p(expected_logit))
    assert verdict.decision is ResidentialDecision.REJECT


def test_weak_mixed_evidence_asks() -> None:
    features = ResidentialFeatures(
        building_tag="apartments",
        building_has_poi_tags=False,
        pois_inside=0,
        landuse_here=(),
        area_m2=None,
        levels=None,
        suite=True,
    )
    verdict = score_residential(features, _CFG)
    expected_logit = _CFG.w_building_residential + _CFG.w_unit_suite
    assert verdict.p_residential == pytest.approx(_logit_to_p(expected_logit))
    assert verdict.decision is ResidentialDecision.ASK


def test_reasons_are_ordered_by_influence_and_carry_their_signed_weight() -> None:
    features = ResidentialFeatures(
        building_tag="house",  # +2.0
        building_has_poi_tags=False,
        pois_inside=0,
        landuse_here=(),
        area_m2=None,
        levels=2,  # +0.5, weaker
        suite=False,
    )
    verdict = score_residential(features, _CFG)
    assert len(verdict.reasons) == 2
    assert "+2.00" in verdict.reasons[0]
    assert "+0.50" in verdict.reasons[1]


def test_a_poi_on_the_building_itself_is_distinguished_from_a_poi_inside_it() -> None:
    on_building = ResidentialFeatures(
        building_tag=None,
        building_has_poi_tags=True,
        pois_inside=0,
        landuse_here=(),
        area_m2=None,
        levels=None,
        suite=False,
    )
    inside = ResidentialFeatures(
        building_tag=None,
        building_has_poi_tags=False,
        pois_inside=3,
        landuse_here=(),
        area_m2=None,
        levels=None,
        suite=False,
    )
    assert "itself carries" in score_residential(on_building, _CFG).reasons[0]
    assert "3 shop" in score_residential(inside, _CFG).reasons[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Suite 200", True),
        ("Ste 4B", True),
        ("ste. 12", True),
        ("12 Oak Street", False),
        ("Suitely Road", False),  # word boundary: not just a prefix match
    ],
)
def test_mentions_suite(text: str, expected: bool) -> None:
    assert mentions_suite(text) is expected
