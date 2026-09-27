"""UrllibTransport: retry policy and error mapping, with urlopen faked out.

Never opens a real socket: ``urllib.request.urlopen`` is monkeypatched to run a
scripted sequence of responses and errors, and ``sleep`` is replaced with a
plain recorder, so a retry test runs in microseconds.
"""

from __future__ import annotations

import email.message
import urllib.error
import urllib.request
from collections.abc import Callable

import pytest

from canopy.errors import GeodataError
from canopy.worldgen.geo.http import UrllibTransport


class _FakeResponse:
    """A minimal stand-in for :class:`http.client.HTTPResponse`."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        """Return the canned body."""
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_urlopen(steps: list[BaseException | _FakeResponse]) -> Callable[..., _FakeResponse]:
    """Return an ``urlopen`` replacement that runs ``steps`` in order, one per call."""
    calls = iter(steps)

    def urlopen(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        del request, timeout
        step = next(calls)
        if isinstance(step, BaseException):
            raise step
        return step

    return urlopen


def _transport(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[BaseException | _FakeResponse],
    *,
    retries: int = 0,
    backoff_s: float = 0.0,
    sleeps: list[float] | None = None,
) -> UrllibTransport:
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(steps))
    record = sleeps if sleeps is not None else []
    return UrllibTransport("canopy-test/1.0", 5.0, retries, backoff_s, sleep=record.append)


def test_get_json_decodes_the_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch, [_FakeResponse(b'{"a": 1}')])
    assert transport.get_json("http://example.test/api", {"q": "x"}) == {"a": 1}


def test_post_form_decodes_the_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch, [_FakeResponse(b'{"ok": true}')])
    assert transport.post_form("http://example.test/api", {"data": "q"}) == {"ok": True}


def test_retries_a_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [
            urllib.error.HTTPError(
                "http://example.test", 429, "Too Many Requests", email.message.Message(), None
            ),
            _FakeResponse(b"{}"),
        ],
        retries=1,
        backoff_s=2.5,
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [2.5]


def test_exhausting_retries_on_a_429_raises_geodata_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(
        monkeypatch,
        [
            urllib.error.HTTPError(
                "http://example.test", 429, "Too Many Requests", email.message.Message(), None
            ),
            urllib.error.HTTPError(
                "http://example.test", 429, "Too Many Requests", email.message.Message(), None
            ),
        ],
        retries=1,
        backoff_s=0.0,
    )
    with pytest.raises(GeodataError, match="429"):
        transport.get_json("http://example.test", {})


def test_a_non_retryable_status_raises_immediately_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [
            urllib.error.HTTPError(
                "http://example.test", 404, "Not Found", email.message.Message(), None
            )
        ],
        retries=3,
        backoff_s=1.0,
        sleeps=sleeps,
    )
    with pytest.raises(GeodataError, match="404"):
        transport.get_json("http://example.test", {})
    assert sleeps == []


def test_a_connection_failure_raises_geodata_error_naming_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport(monkeypatch, [urllib.error.URLError("connection refused")])
    with pytest.raises(GeodataError, match=r"example.test"):
        transport.get_json("http://example.test/api", {})


def test_a_timeout_raises_geodata_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(GeodataError):
        transport.get_json("http://example.test", {})


def test_an_undecodable_body_raises_geodata_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch, [_FakeResponse(b"not json")])
    with pytest.raises(GeodataError):
        transport.get_json("http://example.test", {})


def test_a_non_http_scheme_is_refused_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: calls.append(1))  # noqa: ARG005
    transport = UrllibTransport("canopy-test/1.0", 5.0, 0, 0.0, sleep=lambda _s: None)
    with pytest.raises(GeodataError, match="ftp"):
        transport.get_json("ftp://example.test/api", {})
    assert calls == []


def test_a_timeout_error_then_success_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [TimeoutError("timed out"), _FakeResponse(b"{}")],
        retries=1,
        backoff_s=1.5,
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [1.5]


def test_a_urlerror_wrapping_a_timeout_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [urllib.error.URLError(TimeoutError("timed out")), _FakeResponse(b"{}")],
        retries=1,
        backoff_s=1.0,
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [1.0]


def test_a_plain_connection_error_is_not_retried_even_with_retries_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [urllib.error.URLError("connection refused")],
        retries=3,
        backoff_s=1.0,
        sleeps=sleeps,
    )
    with pytest.raises(GeodataError, match="connection refused"):
        transport.get_json("http://example.test", {})
    assert sleeps == []


def test_backoff_doubles_across_three_retried_504s(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    error = urllib.error.HTTPError(
        "http://example.test", 504, "Gateway Timeout", email.message.Message(), None
    )
    transport = _transport(
        monkeypatch,
        [error, error, error, _FakeResponse(b"{}")],
        retries=3,
        backoff_s=2.0,
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [2.0, 4.0, 8.0]


def test_a_502_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    transport = _transport(
        monkeypatch,
        [
            urllib.error.HTTPError(
                "http://example.test", 502, "Bad Gateway", email.message.Message(), None
            ),
            _FakeResponse(b"{}"),
        ],
        retries=1,
        backoff_s=1.0,
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [1.0]


def test_retry_after_header_overrides_the_backoff_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    headers = email.message.Message()
    headers["Retry-After"] = "7"
    transport = _transport(
        monkeypatch,
        [
            urllib.error.HTTPError("http://example.test", 429, "Too Many Requests", headers, None),
            _FakeResponse(b"{}"),
        ],
        retries=1,
        backoff_s=99.0,  # would be an obviously wrong wait if the header were ignored
        sleeps=sleeps,
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert sleeps == [7.0]


def test_error_message_names_the_attempt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    error = urllib.error.HTTPError(
        "http://example.test", 504, "Gateway Timeout", email.message.Message(), None
    )
    transport = _transport(monkeypatch, [error, error], retries=1, backoff_s=0.0)
    with pytest.raises(GeodataError, match="after 2 attempt"):
        transport.get_json("http://example.test", {})


def test_on_retry_is_told_each_coming_attempt_before_its_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError(
        "http://example.test", 504, "Gateway Timeout", email.message.Message(), None
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", _fake_urlopen([error, TimeoutError(), _FakeResponse(b"{}")])
    )
    events: list[tuple[str, object]] = []
    transport = UrllibTransport(
        "canopy-test/1.0",
        5.0,
        3,
        1.0,
        sleep=lambda s: events.append(("sleep", s)),
        on_retry=lambda nxt, total, wait: events.append(("retry", (nxt, total, wait))),
    )
    assert transport.get_json("http://example.test", {}) == {}
    assert events == [
        ("retry", (2, 4, 1.0)),
        ("sleep", 1.0),
        ("retry", (3, 4, 2.0)),
        ("sleep", 2.0),
    ]
