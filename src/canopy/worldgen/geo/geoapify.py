"""Geoapify: address autocomplete with near-complete coverage, behind an API key.

Photon only knows houses someone has mapped in OpenStreetMap, which in much
of the world is a minority of them, so an address a user types may simply
not be offered. Geoapify's autocomplete adds OpenAddresses and other
authoritative address points to OpenStreetMap, which is what makes the field
behave like the address boxes people know from elsewhere. It was chosen over
the better-known commercial providers for its terms: results may be stored
indefinitely as long as their attribution travels with them, which a
snapshot kept forever needs (Google Places, for one, allows 30 days).

The key comes from the environment variable ``cfg.geoapify_key_env`` names,
never from config, and never reaches a log line: the transport logs hosts
only, and every error it raises names the host, not the URL carrying the key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from canopy.contracts import ResolvedAddress
from canopy.errors import GeodataError
from canopy.worldgen.geo.http import Transport, UrllibTransport
from canopy.worldgen.geo.search import expand_query, matches_house_number, rank, typed_house_number

if TYPE_CHECKING:
    from canopy.config import SiteCfg

__all__ = ["GeoapifyGeocoder"]

#: Below this many characters, a query is not worth a request (or a credit).
_MIN_QUERY_CHARS = 3

#: Candidates requested per suggestion offered: streets and towns come back
#: too and are filtered out, so a few spare keep the list full.
_OVERFETCH = 2

#: Geoapify's spellings of an OpenStreetMap element kind, as used in a ``ref``.
_OSM_TYPE_WORDS = {
    "n": "node",
    "w": "way",
    "r": "relation",
    "node": "node",
    "way": "way",
    "relation": "relation",
}

#: Required by Geoapify's terms on the free plan, alongside the data's own credit.
_GEOAPIFY_CREDIT = "Powered by Geoapify"


def _ref(result: dict[str, Any]) -> str:
    """Give a match a stable id: its OpenStreetMap element when it has one.

    An ``osm:way/<id>`` ref lets the fetch pick that exact building out of
    the map rather than snapping to whatever is nearest the pin; any other
    source's match falls back to Geoapify's own place id.
    """
    datasource = result.get("datasource") or {}
    raw = datasource.get("raw") or {}
    kind = _OSM_TYPE_WORDS.get(str(raw.get("osm_type", "")).casefold())
    if datasource.get("sourcename") == "openstreetmap" and kind and raw.get("osm_id") is not None:
        return f"osm:{kind}/{abs(int(raw['osm_id']))}"
    place_id = result.get("place_id")
    return f"geoapify:{place_id}" if place_id else ""


def _attribution(result: dict[str, Any]) -> str:
    """Geoapify's credit plus the matched record's own source's, as its terms ask."""
    datasource = result.get("datasource") or {}
    source_credit = str(datasource.get("attribution") or "").strip()
    return f"{_GEOAPIFY_CREDIT}; {source_credit}" if source_credit else _GEOAPIFY_CREDIT


def _result_to_address(result: Any, *, provider: str) -> ResolvedAddress | None:
    """One Geoapify result to a :class:`ResolvedAddress`, or ``None`` if it is not a house.

    Raises
    ------
    KeyError, TypeError, ValueError
        If the result is malformed; the caller skips it.
    """
    if not result.get("housenumber"):
        return None
    return ResolvedAddress(
        label=str(result["formatted"]),
        provider=provider,
        ref=_ref(result),
        lat_deg=float(result["lat"]),
        lon_deg=float(result["lon"]),
        attribution=_attribution(result),
    )


class GeoapifyGeocoder:
    """Address search-as-you-type against Geoapify's autocomplete API."""

    name: ClassVar[str] = "geoapify"

    def __init__(self, cfg: SiteCfg, api_key: str, transport: Transport | None = None) -> None:
        """Configure the geocoder.

        Parameters
        ----------
        cfg
            ``worldgen.site`` settings: the endpoint, request identity and
            country filter.
        api_key
            The Geoapify key, read from the environment by the caller.
        transport
            HTTP client. Defaults to a real
            :class:`~canopy.worldgen.geo.http.UrllibTransport` with no
            retries, as for Photon: a keystroke's request is cheaper to drop
            than to hold up typing for.

        Raises
        ------
        GeodataError
            If ``api_key`` is empty.
        """
        if not api_key.strip():
            msg = f"the Geoapify API key is empty; set ${cfg.geoapify_key_env}"
            raise GeodataError(msg)
        self.cfg = cfg
        self._api_key = api_key.strip()
        self.transport: Transport = (
            transport
            if transport is not None
            else UrllibTransport(cfg.user_agent, cfg.timeout_s, 0, 0.0)
        )

    def suggest(self, text: str, *, limit: int) -> list[ResolvedAddress]:
        """Return house addresses matching partly typed ``text``, best first.

        Raises
        ------
        GeodataError
            If Geoapify cannot be reached, refuses the key, or answers with an
            unreadable document.
        """
        query = text.strip()
        if len(query) < _MIN_QUERY_CHARS:
            return []
        params = {
            "text": expand_query(query),
            "limit": str(limit * _OVERFETCH),
            "format": "json",
            "apiKey": self._api_key,
        }
        if self.cfg.suggest_countries:
            codes = ",".join(c.casefold() for c in self.cfg.suggest_countries)
            params["filter"] = f"countrycode:{codes}"
        doc = self.transport.get_json(self.cfg.geoapify_url, params)
        if not isinstance(doc, dict) or not isinstance(doc.get("results"), list):
            msg = (
                "Geoapify returned a document with no usable 'results' list "
                f"(got {type(doc).__name__})"
            )
            raise GeodataError(msg)

        number = typed_house_number(query)
        results: list[ResolvedAddress] = []
        seen: set[str] = set()
        for result in doc["results"]:
            try:
                if number is not None and not matches_house_number(
                    number[0], str(result.get("housenumber", "")), complete=number[1]
                ):
                    continue
                address = _result_to_address(result, provider=self.name)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue  # one malformed result does not spoil the rest
            if address is None or address.label in seen:
                continue
            seen.add(address.label)
            results.append(address)
        return rank(expand_query(query), results, lambda a: a.label)[:limit]
