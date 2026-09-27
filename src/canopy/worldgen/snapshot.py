"""Site snapshots on disk: the frozen evidence an address-built property comes from.

A snapshot is written once, when an address is fetched, and read every time a
property is built from it. Two properties of the format carry the determinism
guarantee:

* Coordinates are rounded to a micrometre *before* the snapshot is used for
  anything (:func:`finalize`), so the in-memory snapshot a fetch returns and
  the one :func:`load_snapshot` reads back are equal to the last bit, and a
  build from either writes the same bytes.
* :attr:`~canopy.contracts.SiteSnapshot.site_id` is a hash of the content
  minus the retrieval times, so fetching unchanged data again names the same
  snapshot rather than minting a new one for every refresh.

Snapshots live under ``out/sites/<site_id>/``, which is gitignored: they hold a
real home address and should not travel with the code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from canopy.contracts import (
    ResidentialDecision,
    ResidentialVerdict,
    ResolvedAddress,
    SiteBuilding,
    SiteSnapshot,
    SourceRecord,
)
from canopy.errors import WorldgenError

__all__ = ["finalize", "load_snapshot", "save_snapshot", "sites_root"]

#: Format version, bumped on any change a reader must know about.
_SCHEMA = 1

#: Decimal places kept in stored coordinates: a micrometre, the OBJ export's
#: precision, and far below anything the map data resolves.
_PRECISION = 6

#: Hex digits of the content hash used as the snapshot's name. Twelve is 48
#: bits: collision-free for any number of sites one machine will ever hold.
_ID_HEX = 12

_FILENAME = "snapshot.json"


def sites_root() -> Path:
    """Directory snapshots are saved under by default, relative to the working directory."""
    return Path("out") / "sites"


def _r(value: float) -> float:
    return round(float(value), _PRECISION)


def _points(arr: Any) -> list[list[float]]:
    return [[_r(v) for v in row] for row in np.asarray(arr, dtype=np.float64).tolist()]


def _building_doc(b: SiteBuilding) -> dict[str, Any]:
    return {
        "footprint": _points(b.footprint),
        "levels": b.levels,
        "height_m": None if b.height_m is None else _r(b.height_m),
        "roof_shape": b.roof_shape,
        "source": b.source,
    }


def _to_doc(s: SiteSnapshot) -> dict[str, Any]:
    """Serialise a snapshot, rounding every coordinate to :data:`_PRECISION`."""
    return {
        "schema": _SCHEMA,
        "site_id": s.site_id,
        "address": {
            "label": s.address.label,
            "provider": s.address.provider,
            "ref": s.address.ref,
            "lat_deg": round(s.address.lat_deg, 8),
            "lon_deg": round(s.address.lon_deg, 8),
        },
        "verdict": {
            "p_residential": round(s.verdict.p_residential, 4),
            "decision": str(s.verdict.decision),
            "reasons": list(s.verdict.reasons),
        },
        "anchor": {"lat_deg": round(s.anchor_lat_deg, 8), "lon_deg": round(s.anchor_lon_deg, 8)},
        "origin_enu_m": [_r(v) for v in np.asarray(s.origin_enu_m).tolist()],
        "north_rad": _r(s.north_rad),
        "lot_m": [_r(s.lot_m[0]), _r(s.lot_m[1])],
        "house": _building_doc(s.house),
        "neighbours": [_building_doc(b) for b in s.neighbours],
        "trees": _points(np.asarray(s.trees).reshape(-1, 3)),
        "poles": _points(np.asarray(s.poles).reshape(-1, 2)),
        "street": None if s.street is None else _points(s.street),
        "street_name": s.street_name,
        "street_width_m": _r(s.street_width_m),
        "aoi": _points(s.aoi),
        "sources": [
            {
                "provider": r.provider,
                "dataset": r.dataset,
                "licence": r.licence,
                "attribution": r.attribution,
                "retrieved_at": r.retrieved_at,
            }
            for r in s.sources
        ],
        "notes": list(s.notes),
    }


def _content_id(doc: dict[str, Any]) -> str:
    """Hash everything that shapes the property: not the id itself, not when it was fetched."""
    content = {k: v for k, v in doc.items() if k != "site_id"}
    content["sources"] = [
        {k: v for k, v in src.items() if k != "retrieved_at"} for src in doc["sources"]
    ]
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:_ID_HEX]


def _building(doc: dict[str, Any]) -> SiteBuilding:
    return SiteBuilding(
        footprint=np.asarray(doc["footprint"], dtype=np.float64).reshape(-1, 2),
        levels=None if doc.get("levels") is None else int(doc["levels"]),
        height_m=None if doc.get("height_m") is None else float(doc["height_m"]),
        roof_shape=None if doc.get("roof_shape") is None else str(doc["roof_shape"]),
        source=str(doc.get("source", "")),
    )


def _from_doc(doc: dict[str, Any]) -> SiteSnapshot:
    if doc.get("schema") != _SCHEMA:
        msg = f"snapshot schema {doc.get('schema')!r} is not the supported {_SCHEMA}"
        raise WorldgenError(msg)
    addr, verdict, anchor = doc["address"], doc["verdict"], doc["anchor"]
    return SiteSnapshot(
        site_id=str(doc["site_id"]),
        address=ResolvedAddress(
            label=str(addr["label"]),
            provider=str(addr["provider"]),
            ref=str(addr["ref"]),
            lat_deg=float(addr["lat_deg"]),
            lon_deg=float(addr["lon_deg"]),
        ),
        verdict=ResidentialVerdict(
            p_residential=float(verdict["p_residential"]),
            decision=ResidentialDecision(verdict["decision"]),
            reasons=tuple(str(r) for r in verdict["reasons"]),
        ),
        anchor_lat_deg=float(anchor["lat_deg"]),
        anchor_lon_deg=float(anchor["lon_deg"]),
        origin_enu_m=np.asarray(doc["origin_enu_m"], dtype=np.float64),
        north_rad=float(doc["north_rad"]),
        lot_m=(float(doc["lot_m"][0]), float(doc["lot_m"][1])),
        house=_building(doc["house"]),
        neighbours=tuple(_building(b) for b in doc["neighbours"]),
        trees=np.asarray(doc["trees"], dtype=np.float64).reshape(-1, 3),
        poles=np.asarray(doc["poles"], dtype=np.float64).reshape(-1, 2),
        street=(
            None
            if doc.get("street") is None
            else np.asarray(doc["street"], dtype=np.float64).reshape(-1, 2)
        ),
        street_name=str(doc.get("street_name", "")),
        street_width_m=float(doc["street_width_m"]),
        aoi=np.asarray(doc["aoi"], dtype=np.float64).reshape(2, 2),
        sources=tuple(
            SourceRecord(
                provider=str(r["provider"]),
                dataset=str(r["dataset"]),
                licence=str(r["licence"]),
                attribution=str(r["attribution"]),
                retrieved_at=str(r["retrieved_at"]),
            )
            for r in doc["sources"]
        ),
        notes=tuple(str(n) for n in doc.get("notes", [])),
    )


def finalize(snapshot: SiteSnapshot) -> SiteSnapshot:
    """Round a freshly fetched snapshot to its stored precision and give it its id.

    Whatever ``site_id`` the input carries is ignored. Build from the returned
    snapshot, never from the input: only the returned one is equal to what
    :func:`load_snapshot` will read back.
    """
    doc = _to_doc(snapshot)
    doc["site_id"] = _content_id(doc)
    return _from_doc(doc)


def save_snapshot(snapshot: SiteSnapshot, out_dir: Path | str | None = None) -> Path:
    """Write ``snapshot.json``.

    Parameters
    ----------
    snapshot
        A snapshot returned by :func:`finalize` (or loaded from disk).
    out_dir
        Directory to write into. Defaults to ``out/sites/<site_id>``.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    WorldgenError
        If the snapshot's id does not match its content, which means it was
        built without :func:`finalize` or edited since.
    """
    doc = _to_doc(snapshot)
    if (expected := _content_id(doc)) != snapshot.site_id:
        msg = (
            f"snapshot id {snapshot.site_id!r} does not match its content ({expected!r}); "
            "pass it through finalize() before saving"
        )
        raise WorldgenError(msg)
    directory = Path(out_dir) if out_dir is not None else sites_root() / snapshot.site_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _FILENAME
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_snapshot(ref: Path | str) -> SiteSnapshot:
    """Read a snapshot by file, by directory, or by bare id under :func:`sites_root`.

    Raises
    ------
    WorldgenError
        If nothing matches ``ref``, the file is not a readable snapshot, or its
        content no longer matches its id (it was edited by hand).
    """
    path = Path(ref)
    if not path.exists() and (sites_root() / str(ref)).is_dir():
        path = sites_root() / str(ref)
    if path.is_dir():
        path = path / _FILENAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = (
            f"no site snapshot at {path} ({exc}); pass a snapshot.json, its directory or a site id"
        )
        raise WorldgenError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise WorldgenError(msg) from exc
    try:
        snapshot = _from_doc(doc)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        msg = f"{path} is missing or malformed: {exc}"
        raise WorldgenError(msg) from exc
    if (expected := _content_id(_to_doc(snapshot))) != snapshot.site_id:
        msg = (
            f"{path} was edited after it was fetched: its content hashes to {expected!r}, "
            f"not its id {snapshot.site_id!r}; fetch the address again"
        )
        raise WorldgenError(msg)
    return snapshot
