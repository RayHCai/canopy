"""The desktop viewer: a headless mapping mission and the window that renders it.

Despite the name, this is not Rerun logging -- ``canopy-view`` opens a native
window (pywebview) whose page draws a live :class:`~canopy.planning.MissionRun`
with three.js; see :mod:`canopy.viz.viewer` and
``docs/adr/0005-desktop-viewer.md`` for why. :class:`ViewerSession` is the pure
Python half: it owns the generated property, the mission run and simulated
time, and can be driven by a test with no display attached. :func:`launch`
is the one place pywebview is imported, lazily, so a machine without the
``viewer`` dependency group still imports :mod:`canopy` cleanly.

Public API: :class:`ViewerSession`, :func:`launch`.
"""

from __future__ import annotations

from canopy.viz.viewer import ViewerSession, launch

__all__ = ["ViewerSession", "launch"]
