"""End to end: a swarm mapping a generated property finds its meter, conduit and bushes.

This is the one test that runs the whole chain -- worldgen, the coloured
ray sensor, the mission and the mapper's detector -- on a real property.
Ground truth appears here only to score the result, never on its way to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from canopy.config import Config
from canopy.contracts import Cls
from canopy.planning import MissionRun
from canopy.sim import load_geometry
from canopy.worldgen import generate_field


@pytest.mark.slow
def test_swarm_finds_the_meter_conduit_and_bushes(cfg: Config, tmp_path: Path) -> None:
    """After a full mission on seed 42 the meter is boxed within 10 cm, with conduit and bushes."""
    manifest = generate_field(42, cfg, out_dir=tmp_path)
    geometry = load_geometry(manifest)
    run = MissionRun(manifest, geometry, 3, cfg, seed=42)
    run.run(cfg.sim.timeout_s)

    found = list(run.mapper.state.discovered.values())
    meters = [o for o in found if o.cls is Cls.METER]
    assert len(meters) == 1
    true_meter = geometry.vertices[manifest.gt_meter_id]
    true_centre = (true_meter.min(axis=0) + true_meter.max(axis=0)) / 2.0
    np.testing.assert_allclose(meters[0].box.center, true_centre, atol=0.1)
    assert any(o.cls is Cls.CONDUIT for o in found)
    assert sum(o.cls is Cls.BUSH for o in found) >= 5
