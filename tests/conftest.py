"""Shared fixtures."""

from __future__ import annotations

import pytest

from canopy.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Return the shipped default configuration, loaded once per session."""
    return load_config()
