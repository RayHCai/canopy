"""The native window: the one place pywebview is imported.

``canopy-view`` opens a native window (pywebview) whose page, served from
:data:`WEB_DIR`, draws a live :class:`~canopy.viz.session.ViewerSession` with
three.js. Why a web renderer in a desktop window, rather than the PyBullet GUI
or Rerun, is recorded in ``docs/adr/0005-desktop-viewer.md``.

:func:`launch` imports pywebview lazily, so a machine without the ``viewer``
dependency group still imports :mod:`canopy` cleanly and only ``canopy-view``
itself fails, with :class:`~canopy.errors.DependencyMissingError`.
:class:`_Bridge` is the whole surface the page can call; everything behind it
is headless and tested without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from canopy.errors import DependencyMissingError
from canopy.viz.models import battery_model, drone_model

if TYPE_CHECKING:
    from canopy.viz.session import ViewerSession

__all__ = ["WEB_DIR", "launch"]

#: The page, its script and its vendored three.js.
WEB_DIR = Path(__file__).resolve().parent / "web"


class _Bridge:
    """What the page may call, as ``window.pywebview.api.<name>``.

    pywebview exposes every public attribute of this object to JavaScript, so it
    is kept deliberately thin and the session itself is never handed over.
    """

    def __init__(self, session: ViewerSession) -> None:
        self._session = session

    def scene(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.scene`."""
        return self._session.scene()

    def world(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.world`."""
        return self._session.world()

    def frame(self, wall_dt_s: float) -> dict[str, Any]:
        """See :meth:`ViewerSession.advance`."""
        return self._session.advance(float(wall_dt_s))

    def set_drone_count(self, n: int) -> dict[str, Any]:
        """See :meth:`ViewerSession.set_drone_count`."""
        return self._session.set_drone_count(int(n))

    def set_time_scale(self, scale: float) -> dict[str, Any]:
        """See :meth:`ViewerSession.set_time_scale`."""
        return self._session.set_time_scale(float(scale))

    def restart(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.restart`."""
        return self._session.restart()

    def new_scene(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.new_scene`."""
        return self._session.new_scene()

    def use_random_location(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.use_random_location`."""
        return self._session.use_random_location()

    def suggest_addresses(self, text: str) -> dict[str, Any]:
        """See :meth:`ViewerSession.suggest_addresses`."""
        return self._session.suggest_addresses(str(text))

    def start_site_build(self, suggestion: dict[str, Any], query: str) -> dict[str, Any]:
        """See :meth:`ViewerSession.start_site_build`."""
        return self._session.start_site_build(suggestion, str(query))

    def confirm_site_build(self, job_id: int) -> dict[str, Any]:
        """See :meth:`ViewerSession.confirm_site_build`."""
        return self._session.confirm_site_build(int(job_id))

    def cancel_site_build(self, job_id: int) -> dict[str, Any]:
        """See :meth:`ViewerSession.cancel_site_build`."""
        return self._session.cancel_site_build(int(job_id))

    def site_job(self, job_id: int) -> dict[str, Any]:
        """See :meth:`ViewerSession.site_job`."""
        return self._session.site_job(int(job_id))

    def run_record(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.run_record`."""
        return self._session.run_record()

    def intake(self) -> dict[str, Any]:
        """See :meth:`ViewerSession.intake`."""
        return self._session.intake()

    def drone_model(self) -> dict[str, Any]:
        """See :func:`drone_model`."""
        return drone_model()

    def battery_model(self) -> dict[str, Any]:
        """See :func:`battery_model`."""
        return battery_model()


def launch(session: ViewerSession, *, debug: bool = False) -> None:
    """Open the Canopy window on ``session`` and block until it is closed.

    Parameters
    ----------
    session
        The mission to render. Its simulation thread is started here and
        stopped when the window closes.
    debug
        Enable the web inspector (right-click, Inspect).

    Raises
    ------
    DependencyMissingError
        If the ``viewer`` dependency group (pywebview) is not installed.
    """
    try:
        import webview  # noqa: PLC0415 - optional dependency, see module docstring
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = "canopy-view needs the `viewer` dependency group: run `uv sync --group viewer`"
        raise DependencyMissingError(msg) from exc

    viewer = session.config.viewer
    session.start()
    webview.create_window(
        "Canopy",
        # A plain path, not a file:// URI: pywebview serves local paths over its
        # built-in HTTP server, and the page's ES-module imports need one.
        url=str(WEB_DIR / "index.html"),
        js_api=_Bridge(session),
        width=viewer.width_px,
        height=viewer.height_px,
        min_size=(720, 480),
        maximized=True,
        background_color="#EEF1F4",
    )
    try:
        webview.start(debug=debug)
    finally:
        session.close()
