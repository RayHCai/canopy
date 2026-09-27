"""HTTP transport for the geo package: the network boundary itself.

Every provider in :mod:`canopy.worldgen.geo` reaches the network through
:class:`Transport` rather than importing ``urllib`` on its own, so a test can
hand it a scripted fake instead of a live socket, and so
``tests/test_architecture.py`` has exactly one place to check for an import of
an HTTP or socket library outside this package.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from canopy.errors import GeodataError
from canopy.log import get_logger

if TYPE_CHECKING:
    from email.message import Message

__all__ = ["Transport", "UrllibTransport"]

_log = get_logger(__name__)

#: HTTP statuses that mean "busy, ask again later" rather than "this request
#: is wrong"; with a timed-out attempt, the only failures worth retrying.
#: Overpass answers 504 when it has no free slot, 429 when this client has
#: used its share, and 502 while a load balancer swaps a backend out.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})

#: Schemes ``urlopen`` may be pointed at. Bandit (ruff S310) flags every
#: dynamic URL-open call since it cannot see this check; validating first is
#: what makes the ignore comments below honest rather than a blanket silence.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


class Transport(Protocol):
    """What a geo provider needs from an HTTP client.

    Narrow on purpose: a fake in tests implements these two methods and
    nothing about ``urllib``, so a provider under test never touches a socket.
    """

    def get_json(self, url: str, params: Mapping[str, str]) -> Any:
        """Return the decoded JSON body of a GET request."""
        ...

    def post_form(self, url: str, form: Mapping[str, str]) -> Any:
        """Return the decoded JSON body of a POST with a urlencoded form body."""
        ...


class UrllibTransport:
    """The real :class:`Transport`, built on the standard library alone.

    Retries a busy answer (429/502/503/504) and an attempt that timed out:
    both are how a loaded public server says "not now", and on the public
    Overpass instance they are routine rather than rare. The wait doubles from
    ``backoff_s`` each time, or is whatever a ``Retry-After`` header asks for,
    and goes through the injectable ``sleep`` so a test never really sleeps.
    Every other failure (a 4xx that means the request itself is wrong, a DNS
    or connection failure, a body that is not JSON) is fatal on the first
    try: retrying it would only spend the request budget on something that
    will not change.
    """

    def __init__(
        self,
        user_agent: str,
        timeout_s: float,
        retries: int,
        backoff_s: float,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """Configure the client.

        Parameters
        ----------
        user_agent
            Sent as ``User-Agent``; both Photon and Overpass expect one.
        timeout_s
            Socket timeout for one attempt.
        retries
            Extra attempts after a busy answer or a timeout, beyond the first.
        backoff_s
            Wait before the first retry; each later one waits twice as long.
        sleep
            Replaces :func:`time.sleep`; a test passes a no-op recorder.
        on_retry
            Called as ``on_retry(next_attempt, attempts, wait_s)`` before each
            wait, so a caller can say "busy, retrying" rather than look hung
            through a minute of backoff.
        """
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._retries = retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._on_retry = on_retry

    def get_json(self, url: str, params: Mapping[str, str]) -> Any:
        """Return the decoded JSON body of a GET request."""
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}" if query else url
        return self._request(full_url, data=None)

    def post_form(self, url: str, form: Mapping[str, str]) -> Any:
        """Return the decoded JSON body of a POST with a urlencoded form body."""
        body = urllib.parse.urlencode(form).encode("ascii")
        return self._request(url, data=body)

    def _request(self, url: str, *, data: bytes | None) -> Any:
        """Send one request, retrying a busy answer, and decode its JSON body.

        Never logs ``url``: its query string can carry a home address, and a
        log line is not the place for one. Only the host is logged, at DEBUG.
        """
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in _ALLOWED_SCHEMES:
            msg = f"refusing to fetch a {parts.scheme!r} URL for host {parts.netloc}"
            raise GeodataError(msg)
        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        # Scheme is validated above; only http(s) ever reaches urlopen.
        request = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310

        attempts = self._retries + 1
        for attempt in range(attempts):
            _log.debug("requesting %s", parts.netloc)
            try:
                # Scheme is validated above; only http(s) ever reaches urlopen.
                with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRYABLE_STATUSES and attempt + 1 < attempts:
                    self._back_off(attempt, attempts, self._wait_s(attempt, exc.headers))
                    continue
                msg = (
                    f"{parts.netloc} answered HTTP {exc.code} ({exc.reason}) "
                    f"after {attempt + 1} attempt(s)"
                )
                raise GeodataError(msg) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                if _is_timeout(reason) and attempt + 1 < attempts:
                    self._back_off(attempt, attempts, self._wait_s(attempt, None))
                    continue
                msg = f"could not reach {parts.netloc} after {attempt + 1} attempt(s): {reason}"
                raise GeodataError(msg) from exc
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                msg = f"{parts.netloc} returned a body that could not be decoded as JSON: {exc}"
                raise GeodataError(msg) from exc
        # Unreachable for retries >= 0 (every branch above returns or raises);
        # kept so a misconfigured negative retry count fails loudly rather
        # than silently returning nothing.
        msg = f"{parts.netloc} gave up after {attempts} attempt(s)"
        raise GeodataError(msg)

    def _back_off(self, attempt: int, attempts: int, wait_s: float) -> None:
        """Report the coming retry, then wait ``wait_s`` before it."""
        if self._on_retry is not None:
            self._on_retry(attempt + 2, attempts, wait_s)
        self._sleep(wait_s)

    def _wait_s(self, attempt: int, headers: Message | None) -> float:
        """Seconds to wait before retry number ``attempt + 1``.

        A ``Retry-After`` in whole seconds is the server's own estimate and
        wins; the HTTP-date form is rare enough from these services that it
        falls back to the doubling backoff rather than being parsed.
        """
        retry_after = headers.get("Retry-After") if headers is not None else None
        if retry_after is not None and retry_after.strip().isdigit():
            return float(retry_after.strip())
        return self._backoff_s * 2.0**attempt


def _is_timeout(reason: object) -> bool:
    """Whether a failed attempt timed out, however ``urllib`` chose to wrap it.

    A read timeout surfaces as a bare :class:`TimeoutError`; a connect timeout
    arrives as a :class:`urllib.error.URLError` whose ``reason`` is one.
    """
    return isinstance(reason, TimeoutError)
