"""Photon: address search-as-you-type, built on OpenStreetMap.

Nominatim's public instance forbids autocomplete, so suggestions as a user
types come from Photon instead: free, keyless, and limited to the houses
someone has mapped in OpenStreetMap -- the fallback when no Geoapify key is
configured (:mod:`canopy.worldgen.geo.geoapify`). Its GeoJSON is looser than
Overpass's: a feature can be a street, a town or an address, and only the
last is useful here, so :meth:`PhotonGeocoder.suggest` filters and reshapes
rather than just parsing, and re-ranks with :mod:`canopy.worldgen.geo.search`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from canopy.contracts import ResolvedAddress
from canopy.errors import GeodataError
from canopy.worldgen.geo.http import Transport, UrllibTransport
from canopy.worldgen.geo.search import expand_query, matches_house_number, rank, typed_house_number

if TYPE_CHECKING:
    from canopy.config import SiteCfg

__all__ = ["PhotonGeocoder"]

#: Below this many characters, a query is not worth a request.
_MIN_QUERY_CHARS = 3

#: Candidates requested per suggestion offered. Streets, towns and houses
#: with the wrong number are filtered back out, and the re-rank needs the
#: right house to be somewhere in the list to promote it.
_OVERFETCH = 3

#: Photon's single-letter OSM element kinds, as used in a ``ref``.
_OSM_TYPE_WORDS = {"N": "node", "W": "way", "R": "relation"}


def _label(props: dict[str, Any]) -> str:
    """Build one display line from Photon's properties, omitting missing parts.

    E.g. ``"12 Oak Street, Springfield, Illinois 62704, United States"``, or a
    shorter line wherever a part (a street, a postcode, ...) is not mapped.
    """
    street_part = " ".join(str(p) for p in (props.get("housenumber"), props.get("street")) if p)
    place = props.get("city") or props.get("town") or props.get("district")
    state_zip = " ".join(str(p) for p in (props.get("state"), props.get("postcode")) if p)
    country = props.get("country")
    return ", ".join(str(p) for p in (street_part, place, state_zip, country) if p)


def _feature_to_address(feature: Any, *, provider: str) -> ResolvedAddress | None:
    """One GeoJSON feature to a :class:`ResolvedAddress`, or ``None`` if it is not an address.

    Raises
    ------
    KeyError, TypeError, ValueError
        If the feature is malformed. The caller treats every one of these as
        "skip this feature", so they are not caught here.
    """
    props = feature["properties"]
    if not props.get("housenumber"):
        return None
    lon, lat = feature["geometry"]["coordinates"]
    osm_type = _OSM_TYPE_WORDS[props["osm_type"]]
    ref = f"osm:{osm_type}/{int(props['osm_id'])}"
    return ResolvedAddress(
        label=_label(props), provider=provider, ref=ref, lat_deg=float(lat), lon_deg=float(lon)
    )


class PhotonGeocoder:
    """Address search-as-you-type against a Photon instance."""

    name: ClassVar[str] = "photon"

    def __init__(self, cfg: SiteCfg, transport: Transport | None = None) -> None:
        """Configure the geocoder.

        Parameters
        ----------
        cfg
            ``worldgen.site`` settings: the endpoint and request identity.
        transport
            HTTP client. Defaults to a real :class:`~canopy.worldgen.geo.http.UrllibTransport`
            with no retries: a busy keystroke request is cheaper to drop than
            to hold up typing for.
        """
        self.cfg = cfg
        self.transport: Transport = (
            transport
            if transport is not None
            else UrllibTransport(cfg.user_agent, cfg.timeout_s, 0, 0.0)
        )

    def suggest(self, text: str, *, limit: int) -> list[ResolvedAddress]:
        """Return addresses matching partly typed ``text``, best first.

        Sends the query with its directional abbreviations spelled out,
        over-fetches, keeps only houses -- with the typed house number, in
        ``cfg.suggest_countries`` when that is set -- and re-ranks them by how
        many typed words each contains before capping at ``limit``.

        Raises
        ------
        GeodataError
            If Photon cannot be reached or answers with an unreadable document.
        """
        query = text.strip()
        if len(query) < _MIN_QUERY_CHARS:
            return []
        params = {"q": expand_query(query), "limit": str(limit * _OVERFETCH)}
        doc = self.transport.get_json(self.cfg.geocoder_url, params)
        if not isinstance(doc, dict) or not isinstance(doc.get("features"), list):
            msg = (
                "Photon returned a document with no usable 'features' list "
                f"(got {type(doc).__name__})"
            )
            raise GeodataError(msg)

        number = typed_house_number(query)
        countries = {c.casefold() for c in self.cfg.suggest_countries}
        results: list[ResolvedAddress] = []
        seen: set[str] = set()
        for feature in doc["features"]:
            try:
                props = feature["properties"]
                if countries and str(props.get("countrycode", "")).casefold() not in countries:
                    continue
                if number is not None and not matches_house_number(
                    number[0], str(props.get("housenumber", "")), complete=number[1]
                ):
                    continue
                address = _feature_to_address(feature, provider=self.name)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue  # one malformed feature does not spoil the rest
            if address is None or address.label in seen:
                continue
            seen.add(address.label)
            results.append(address)
        return rank(expand_query(query), results, lambda a: a.label)[:limit]
