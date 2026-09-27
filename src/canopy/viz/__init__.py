"""The desktop viewer: a headless mapping mission and the window that renders it.

Despite the name, this is not Rerun logging -- ``canopy-view`` opens a native
window (pywebview) whose page draws a live :class:`~canopy.planning.MissionRun`
with three.js; see ``docs/adr/0005-desktop-viewer.md`` for why.
:class:`ViewerSession` (:mod:`canopy.viz.session`) is the pure Python half: it
owns the generated property, the mission run and simulated time, and can be
driven by a test with no display attached. Its address builds run in
:mod:`canopy.viz.site_jobs`, its per-tick snapshots and run log live in
:mod:`canopy.viz.recording`, and the drone and battery meshes the page draws
are read by :mod:`canopy.viz.models`. :func:`launch` (:mod:`canopy.viz.bridge`)
is the one place pywebview is imported, lazily, so a machine without the
``viewer`` dependency group still imports :mod:`canopy` cleanly.

Public API: :class:`ViewerSession`, :func:`launch`.
"""

from __future__ import annotations

from canopy.viz.bridge import launch
from canopy.viz.session import ViewerSession

__all__ = ["ViewerSession", "launch"]
