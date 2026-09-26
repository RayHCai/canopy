# 2. Python 3.11 with uv + pyenv, not conda

Date: 2026-09-25
Status: accepted

## Context

`spec.md` specifies Python 3.10 in a conda-forge environment, because
conda-forge ships a prebuilt PyBullet and so avoids an MSVC compile on Windows.
On the actual dev machine there is no conda: the Python workflow is pyenv-win,
and uv is already installed.

Surveying wheel availability settled the version:

| | 3.10 | 3.11 | 3.12 |
|---|---|---|---|
| open3d, rerun-sdk, shapely, trimesh, mapbox-earcut | yes | yes | yes |
| scipy | 1.15 | **1.17** | 1.18 |
| scikit-image | 0.25 | **0.26** | 0.26 |
| numpy | 2.2 | **2.4** | 2.5 |
| pybullet (Linux wheel) | yes | **yes** | **no wheel at all** |

PyBullet's newest CPython wheel is cp311, and it is manylinux-only. On 3.12 it
must be compiled from sdist on *every* platform.

`spec.md` also advises `pip install "numpy<2"` for an ABI error. That guidance is
stale: upstream gym-pybullet-drones now *requires* numpy >= 2.5, and nothing in
the stack needs numpy 1.x.

## Decision

Python 3.11.9, pinned in `.python-version` (honoured by both pyenv and uv).
Dependencies declared in `pyproject.toml` and locked in `uv.lock`. No conda, and
no `numpy<2` pin.

## Consequences

- Nothing compiles from source on any platform, on either OS.
- `.python-version` serves pyenv and uv at once, so the user's existing workflow
  is unchanged.
- `uv.lock` is committed and is the reproducibility guarantee; `environment.yml`
  is not carried.
- Moving to 3.12 later means either a from-source PyBullet build or dropping the
  physics track.
