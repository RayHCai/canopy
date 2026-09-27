"""expand_query, typed_house_number, matches_house_number and rank: pure text helpers."""

from __future__ import annotations

import pytest

from canopy.worldgen.geo.search import (
    expand_query,
    matches_house_number,
    rank,
    typed_house_number,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("339 N Oak Park", "339 North Oak Park"),
        ("1600 Pennsylvania Ave NW", "1600 Pennsylvania Ave Northwest"),
        ("339 N", "339 N"),
        ("12 E. Main St, Springfield", "12 East Main St, Springfield"),
    ],
)
def test_expand_query(text: str, expected: str) -> None:
    assert expand_query(text) == expected


def test_expand_query_never_expands_st() -> None:
    """'St' is ambiguous (Street or Saint), so it is left for the geocoder to match on its own."""
    assert expand_query("339 St Oak") == "339 St Oak"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("339 N Oak", ("339", True)),
        ("33", ("33", False)),
        ("Oak Park", None),
        ("", None),
    ],
)
def test_typed_house_number(text: str, expected: tuple[str, bool] | None) -> None:
    assert typed_house_number(text) == expected


@pytest.mark.parametrize(
    ("housenumber", "expected"),
    [
        ("339", True),
        ("339A", True),
        ("339 - 341", True),
        ("3392", False),
        ("12161", False),
    ],
)
def test_matches_house_number_complete(housenumber: str, expected: bool) -> None:
    assert matches_house_number("339", housenumber, complete=True) is expected


def test_matches_house_number_incomplete_matches_an_extension() -> None:
    assert matches_house_number("33", "339", complete=False) is True


def test_rank_promotes_the_label_matching_more_typed_words() -> None:
    labels = [
        "339 Oak Park Road, Oak Park, Nova Scotia",
        "339 North Oak Street, Summit Hill",
        "339 North Oak Park Avenue, Oak Park, Illinois",
    ]
    ranked = rank("339 N Oak Pa", labels, lambda label: label)
    assert ranked[0] == "339 North Oak Park Avenue, Oak Park, Illinois"


def test_rank_is_stable_on_ties() -> None:
    labels = ["1 Oak Street", "2 Oak Street", "3 Oak Street"]
    assert rank("Oak", labels, lambda label: label) == labels


def test_rank_with_no_words_beyond_the_number_returns_input_order() -> None:
    labels = ["339 Oak Park Road", "339 North Oak Park Avenue"]
    assert rank("339", labels, lambda label: label) == labels


def test_rank_matches_an_abbreviated_street_type_against_its_expansion() -> None:
    labels = ["1 Oak Road", "1 Oak Avenue"]
    ranked = rank("1 Oak Ave", labels, lambda label: label)
    assert ranked[0] == "1 Oak Avenue"


def test_rank_breaks_a_tie_by_matches_in_the_street_part() -> None:
    # "Pa" matches both "Park" and the state "PA"; the street part decides.
    labels = [
        "339 North Oak Street, Summit Hill, PA 18250, United States",
        "339 North Oak Park Avenue, Oak Park, IL 60302, United States",
    ]
    assert rank("339 N Oak Pa", labels, str)[0].startswith("339 North Oak Park Avenue")
