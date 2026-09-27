"""Address text as a person types it: normalizing the query, ranking the matches.

A geocoder ranks candidates by its own notion of relevance, which does not
know how a street address is typed. Photon reads "339 N Oak Park" as five
loose words, so Nova Scotia's "339 Oak Park Road" outranks Illinois's "339
North Oak Park Avenue", and a house number is just one more word to it, so
"12161 FGCU Lake Parkway" can turn up for "339". These helpers put the
conventions back, for every provider alike: spell out the directional
abbreviations the geocoder does not, insist that a typed house number is the
one matched, and re-rank by how many of the typed words a candidate contains.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["expand_query", "matches_house_number", "rank", "typed_house_number"]

_T = TypeVar("_T")

#: Directional abbreviations, spelled out before a query is sent: the one
#: kind of abbreviation measured to change Photon's ranking ("339 N Oak Park"
#: puts two Canadian houses first; "339 North Oak Park" puts the right one
#: first). Street types ("Ave", "Rd") it already matches on its own.
_DIRECTIONS = {
    "n": "North",
    "s": "South",
    "e": "East",
    "w": "West",
    "ne": "Northeast",
    "nw": "Northwest",
    "se": "Southeast",
    "sw": "Southwest",
}

#: What a typed abbreviation may stand for, for ranking only. Never sent to a
#: geocoder: "St" is as often "Saint" as "Street", and guessing wrong would
#: lose the right match rather than just misorder it.
_EXPANSIONS: dict[str, frozenset[str]] = {
    **{short: frozenset({full.casefold()}) for short, full in _DIRECTIONS.items()},
    "st": frozenset({"street", "saint"}),
    "ave": frozenset({"avenue"}),
    "av": frozenset({"avenue"}),
    "rd": frozenset({"road"}),
    "dr": frozenset({"drive"}),
    "ln": frozenset({"lane"}),
    "blvd": frozenset({"boulevard"}),
    "ct": frozenset({"court"}),
    "pl": frozenset({"place"}),
    "hwy": frozenset({"highway"}),
    "pkwy": frozenset({"parkway"}),
    "ter": frozenset({"terrace"}),
    "cir": frozenset({"circle"}),
    "sq": frozenset({"square"}),
    "mt": frozenset({"mount"}),
    "ft": frozenset({"fort"}),
}

#: A two-letter direction is unambiguous even as the last, still-being-typed
#: word; a single letter there is more likely the start of a street name.
_UNAMBIGUOUS_DIRECTION_LEN = 2

#: A house number: digits first, then whatever a numbering scheme appends
#: ("12A", "33-35", "1/2").
_HOUSE_NUMBER_RE = re.compile(r"\d[\w/-]*")
_WORD_RE = re.compile(r"\w+")


def expand_query(text: str) -> str:
    """Spell out directional abbreviations ("N", "NW") in a typed address.

    The last word is left alone when it is a single letter: "339 N" is as
    likely the start of "339 Newport" as "339 North", and a prefix is what
    search-as-you-type is still matching on.

    Examples
    --------
    >>> expand_query("339 N Oak Park")
    '339 North Oak Park'
    >>> expand_query("1600 Pennsylvania Ave NW")
    '1600 Pennsylvania Ave Northwest'
    >>> expand_query("339 N")
    '339 N'
    """
    words = text.split()
    for i, word in enumerate(words):
        core = word.rstrip(",.")
        full = _DIRECTIONS.get(core.casefold())
        if full is None:
            continue
        is_last = i == len(words) - 1
        if is_last and len(core) < _UNAMBIGUOUS_DIRECTION_LEN:
            continue
        # An abbreviation's own full stop goes with it ("E." -> "East");
        # a comma after it still separates what follows.
        words[i] = full + word[len(core) :].replace(".", "")
    return " ".join(words)


def typed_house_number(text: str) -> tuple[str, bool] | None:
    """Split off the house number a query starts with, and say whether it is finished.

    Returns
    -------
    tuple[str, bool] | None
        The number, casefolded, and ``True`` once another word follows it (so
        no more digits are coming); ``None`` if the query does not start with
        a number.
    """
    words = text.split()
    if not words:
        return None
    first = words[0].rstrip(",").casefold()
    if not _HOUSE_NUMBER_RE.fullmatch(first):
        return None
    return first, len(words) > 1


def matches_house_number(typed: str, housenumber: str, *, complete: bool) -> bool:
    """Whether a candidate's house number is the one typed (so far).

    While the number is still being typed any extension matches: "33" keeps
    "339" in play. Once it is ``complete``, a candidate may only extend it
    with a non-digit: "339" keeps "339A" and "339-341" but rules out "3392".
    """
    candidate = housenumber.casefold().replace(" ", "")
    if not candidate.startswith(typed):
        return False
    return not complete or not candidate[len(typed) : len(typed) + 1].isdigit()


def rank(query: str, results: Sequence[_T], label_of: Callable[[_T], str]) -> list[_T]:
    """Order ``results`` by how many typed words each label contains, most first.

    The sort is stable, so the geocoder's own order breaks ties; this only
    ever promotes a candidate that matches more of what was typed. A typed
    word counts if a label word equals it or is one of its expansions ("Ave"
    for "Avenue"). Only the last word may match as a prefix, since it is the
    one still being typed: letting every word do so would have "N" match
    "Nova Scotia" as readily as "North".

    Ties on that count go to the candidate matching more of them in its
    street part (the label up to its first comma): "339 N Oak Pa" is headed
    for "339 North Oak Park Avenue", not for a "339 North Oak Street" whose
    "Pa" is only its state, Pennsylvania.
    """
    typed = [w for w in _WORD_RE.findall(query.casefold()) if not w[:1].isdigit()]
    if not typed:
        return list(results)
    last = len(typed) - 1

    def matched(words: list[str]) -> int:
        return sum(
            any(
                w == t or w in _EXPANSIONS.get(t, ()) or (i == last and w.startswith(t))
                for w in words
            )
            for i, t in enumerate(typed)
        )

    def score(result: _T) -> tuple[int, int]:
        label = label_of(result).casefold()
        street = label.split(",", 1)[0]
        return matched(_WORD_RE.findall(label)), matched(_WORD_RE.findall(street))

    return sorted(results, key=score, reverse=True)
