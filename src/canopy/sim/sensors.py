"""The 360-degree ray sensor.

:class:`RaySensor` wraps an Open3D ``RaycastingScene`` built once from a
:class:`~canopy.contracts.SceneGeometry` and casts an equirectangular grid of
rays from a drone's position each sensor tick. Open3D is a core dependency
(not one of the ``rl``/``detect``/``viewer`` extras), so the import happens at
module scope rather than lazily inside a function.

Each hit also reports a colour, as a camera co-registered with the ranger
would: the true colour of the object hit, lit by a fixed sun with Lambertian
shading, plus per-channel noise. That colour and the range are all perception
gets (:meth:`~canopy.contracts.Scan.observation`); the object and triangle ids
stay with the simulator's coverage bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import open3d as o3d

from canopy.contracts import Scan
from canopy.errors import SimulationError

if TYPE_CHECKING:
    from canopy.config import SensorCfg
    from canopy.contracts import Points, SceneGeometry, Vec3

__all__ = ["RaySensor", "build_raycasting_scene", "sky_exposed"]

#: Upper-hemisphere probe directions for :func:`sky_exposed`: the zenith plus a
#: ring of eight at 45 degrees elevation. Nine rays are enough to tell an
#: enclosed face (every one hits the roof or a wall) from an open one (the
#: zenith clears a lawn, and some slant ray clears a wall under the eaves).
_SKY_ELEVATION_RAD = np.pi / 4.0
_SKY_RING = 8
#: How far off a surface a sky probe starts, so it cannot hit its own triangle.
_SKY_OFFSET_M = 0.03
#: Rays cast per Open3D call in :func:`sky_exposed`, which bounds peak memory.
_SKY_CHUNK = 1 << 19
#: Colour of every object in hand-built test geometry, which has none of its own.
_UNCOLOURED = 128.0
#: Largest value of an 8-bit colour channel.
_CHANNEL_MAX = 255


def build_raycasting_scene(
    vertices: Sequence[Points], faces: Sequence[npt.NDArray[np.int32]]
) -> Any:
    """Build an Open3D raycasting scene whose geometry ids equal list positions.

    Raises
    ------
    SimulationError
        If Open3D hands out geometry ids in any other order. Everything
        downstream turns a geometry id straight into an ``obj_id``.
    """
    scene = o3d.t.geometry.RaycastingScene()
    for obj_id, (verts, tris) in enumerate(zip(vertices, faces, strict=True)):
        verts_t = o3d.core.Tensor.from_numpy(np.ascontiguousarray(verts, dtype=np.float32))
        faces_t = o3d.core.Tensor.from_numpy(np.ascontiguousarray(tris, dtype=np.uint32))
        geom_id = int(scene.add_triangles(verts_t, faces_t))
        if geom_id != obj_id:
            msg = (
                f"Open3D assigned geometry id {geom_id} to object {obj_id}; "
                "the scene relies on add_triangles returning ids in call order"
            )
            raise SimulationError(msg)
    return scene


def sky_exposed(scene: Any, centroids: Points, normals: Points) -> npt.NDArray[np.bool_]:
    """Whether open sky is visible from just off either side of each triangle.

    This is the test for "could a drone outside ever see this face?". Authored
    house shells carry their inner wall faces and interior partitions, which
    is most of the WALL area, and no viewpoint outside the house reaches them.
    Counting them toward coverage would cap it far below any completion
    threshold. A face is exposed when any of nine upward probes, started a
    few centimetres off either side, escapes the scene without a hit.

    Parameters
    ----------
    scene
        From :func:`build_raycasting_scene`.
    centroids, normals
        Per-triangle centroid and unit normal, shape ``(T, 3)``. A zero normal
        probes from the centroid itself.

    Returns
    -------
    numpy.ndarray
        Shape ``(T,)``.
    """
    az = 2.0 * np.pi * np.arange(_SKY_RING) / _SKY_RING
    ring = np.column_stack(
        [
            np.cos(_SKY_ELEVATION_RAD) * np.cos(az),
            np.cos(_SKY_ELEVATION_RAD) * np.sin(az),
            np.full(_SKY_RING, np.sin(_SKY_ELEVATION_RAD)),
        ]
    )
    probes = np.vstack([[0.0, 0.0, 1.0], ring])
    starts = np.concatenate(
        [centroids + _SKY_OFFSET_M * normals, centroids - _SKY_OFFSET_M * normals]
    )
    n_tri, n_probe = len(centroids), len(probes)
    rays = np.empty((len(starts) * n_probe, 6), dtype=np.float32)
    rays[:, :3] = np.repeat(starts, n_probe, axis=0)
    rays[:, 3:] = np.tile(probes, (len(starts), 1))

    escaped = np.empty(len(rays), dtype=np.bool_)
    for lo in range(0, len(rays), _SKY_CHUNK):
        chunk = o3d.core.Tensor.from_numpy(np.ascontiguousarray(rays[lo : lo + _SKY_CHUNK]))
        t_hit = np.asarray(scene.cast_rays(chunk)["t_hit"].numpy())
        escaped[lo : lo + _SKY_CHUNK] = ~np.isfinite(t_hit)
    per_start: npt.NDArray[np.bool_] = np.any(escaped.reshape(2 * n_tri, n_probe), axis=1)
    exposed: npt.NDArray[np.bool_] = per_start[:n_tri] | per_start[n_tri:]
    return exposed


class RaySensor:
    """A 360-degree ray sensor cast against a fixed :class:`SceneGeometry`.

    Built once per scene: constructing the Open3D BVH is the expensive part,
    scanning from it is cheap, and the sim calls :meth:`scan` once per drone
    per sensor tick.
    """

    def __init__(self, geometry: SceneGeometry, cfg: SensorCfg, rng: np.random.Generator) -> None:
        self._geometry = geometry
        self._cfg = cfg
        self._rng = rng
        # Colour noise draws from its own child stream, so adding the colour
        # channel left every range sample and scan jitter -- and with them the
        # flown missions -- exactly as they were.
        self._colour_rng = rng.spawn(1)[0]

        self._scene = build_raycasting_scene(geometry.vertices, geometry.faces)

        n_objects = len(geometry.vertices)
        self._obj_rgb = (
            np.full((n_objects, 3), _UNCOLOURED)
            if geometry.obj_color is None
            else np.asarray(geometry.obj_color, dtype=np.float64).reshape(n_objects, 3)
        )
        sun = np.asarray(cfg.sun_dir, dtype=np.float64)
        self._sun = sun / np.linalg.norm(sun)
        self._sky = np.asarray(cfg.sky_rgb, dtype=np.uint8)

        self._dirs = _direction_grid(cfg.az_rays, cfg.el_rays, cfg.el_min_deg, cfg.el_max_deg)
        # Azimuths repeat every 2*pi/az_rays; jittering by less than that keeps
        # the grid's elevation banding intact while still sampling new surface
        # points between scans (see scan()).
        self._az_step = 2.0 * np.pi / cfg.az_rays

    @property
    def dirs(self) -> Points:
        """The un-jittered direction grid, shape ``(n_rays, 3)``, unit vectors."""
        return self._dirs

    @property
    def n_rays(self) -> int:
        """Rays per scan."""
        return self._cfg.n_rays

    def scan(self, drone_id: int, t: float, origin: Vec3) -> Scan:
        """Cast one 360-degree scan from ``origin``.

        Parameters
        ----------
        drone_id
            Owning drone, carried through to the returned :class:`Scan`.
        t
            Sim time of the scan, carried through unchanged.
        origin
            World-space ray origin, shape ``(3,)``.

        Returns
        -------
        Scan
            Hits with ``t_hit`` beyond ``max_range_m`` are misses. Range noise
            (Gaussian, sigma ``range_noise_m``) is added to hit distances when
            configured, then clamped to non-negative. ``rgb`` holds each hit's
            shaded colour and the sky colour for each miss.

        Notes
        -----
        The direction grid is rotated about Z by a random azimuth offset drawn
        fresh each call. Repeated scans from one spot would otherwise hit the
        exact same points every time, which starves triangle coverage (the
        reveal and the ground-band metric) of new samples between moves.
        """
        jitter = float(self._rng.uniform(0.0, self._az_step))
        cos_j, sin_j = np.cos(jitter), np.sin(jitter)
        rot = np.array([[cos_j, -sin_j, 0.0], [sin_j, cos_j, 0.0], [0.0, 0.0, 1.0]])
        dirs = self._dirs @ rot.T

        origin_arr = np.asarray(origin, dtype=np.float64)
        rays_np = np.concatenate([np.broadcast_to(origin_arr, dirs.shape), dirs], axis=1).astype(
            np.float32
        )
        rays = o3d.core.Tensor.from_numpy(np.ascontiguousarray(rays_np))

        ans: dict[str, Any] = self._scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy().astype(np.float64)
        geometry_ids = ans["geometry_ids"].numpy().astype(np.int64)
        primitive_ids = ans["primitive_ids"].numpy().astype(np.int64)

        hit = np.isfinite(t_hit) & (t_hit <= self._cfg.max_range_m)

        dist = np.full(dirs.shape[0], np.inf, dtype=np.float64)
        obj_ids = np.full(dirs.shape[0], -1, dtype=np.int32)
        tri_ids = np.full(dirs.shape[0], -1, dtype=np.int32)

        dist[hit] = t_hit[hit]
        if self._cfg.range_noise_m > 0.0:
            noise = self._rng.normal(0.0, self._cfg.range_noise_m, size=int(hit.sum()))
            dist[hit] = np.clip(dist[hit] + noise, 0.0, None)
        obj_ids[hit] = geometry_ids[hit].astype(np.int32)
        tri_ids[hit] = (
            self._geometry.obj_tri_offset[geometry_ids[hit]] + primitive_ids[hit]
        ).astype(np.int32)

        rgb = np.empty((dirs.shape[0], 3), dtype=np.uint8)
        rgb[:] = self._sky
        rgb[hit] = self._shade(obj_ids[hit], tri_ids[hit], dirs[hit])

        return Scan(
            drone_id=drone_id,
            t=t,
            origin=origin_arr,
            dirs=dirs,
            dist=dist,
            obj_ids=obj_ids,
            tri_ids=tri_ids,
            rgb=rgb,
        )

    def _shade(
        self, obj_ids: npt.NDArray[np.int32], tri_ids: npt.NDArray[np.int32], dirs: Points
    ) -> npt.NDArray[np.uint8]:
        """Return the colour a camera would record for each hit, shape ``(K, 3)``.

        ``ambient + (1 - ambient) * max(0, n . sun)`` times the object's true
        colour, plus Gaussian noise, rounded to 8 bits. Authored art winds its
        triangles inconsistently, so each normal is first turned to face the
        ray that found it: the lit side is the side the drone can see.
        """
        normals = self._geometry.tri_normal[tri_ids]
        toward = -np.sign(np.einsum("ij,ij->i", normals, dirs))
        lambert = np.clip((normals @ self._sun) * toward, 0.0, None)
        shade = self._cfg.ambient + (1.0 - self._cfg.ambient) * lambert
        colour = self._obj_rgb[obj_ids] * shade[:, None]
        if self._cfg.rgb_noise > 0.0:
            colour += self._colour_rng.normal(0.0, self._cfg.rgb_noise, size=colour.shape)
        shaded: npt.NDArray[np.uint8] = np.clip(np.rint(colour), 0, _CHANNEL_MAX).astype(np.uint8)
        return shaded

    def surface_distance(self, points: Points) -> npt.NDArray[np.float64]:
        """Unsigned distance from each point to the nearest scene surface.

        Ground truth for tests and evaluation collision checks -- the planner
        never reads this, it only ever sees the mapped occupancy grid.

        Parameters
        ----------
        points
            Query points, shape ``(N, 3)``.

        Returns
        -------
        numpy.ndarray
            Distance in metres, shape ``(N,)``.
        """
        pts = o3d.core.Tensor.from_numpy(np.ascontiguousarray(points, dtype=np.float32))
        return np.asarray(self._scene.compute_distance(pts).numpy(), dtype=np.float64)


def _direction_grid(az_rays: int, el_rays: int, el_min_deg: float, el_max_deg: float) -> Points:
    """Build the equirectangular grid of unit ray directions, Z-up.

    Azimuth spans ``[0, 2*pi)`` (half-open: 0 and 2*pi are the same ray, so the
    endpoint is dropped) with ``az_rays`` samples; elevation spans
    ``[el_min_deg, el_max_deg]`` inclusive with ``el_rays`` samples. The
    ``(az, el)`` grid is flattened azimuth-major, an ordering convention
    private to this module and undone nowhere -- callers only ever see
    ``(n_rays, 3)``.
    """
    az = np.linspace(0.0, 2.0 * np.pi, az_rays, endpoint=False)
    el = np.deg2rad(np.linspace(el_min_deg, el_max_deg, el_rays))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    az_flat = az_grid.ravel()
    el_flat = el_grid.ravel()
    x = np.cos(el_flat) * np.cos(az_flat)
    y = np.cos(el_flat) * np.sin(az_flat)
    z = np.sin(el_flat)
    return np.stack([x, y, z], axis=1)
