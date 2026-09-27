"""Real-world site data: the only package in Canopy that touches the network.

Everything here runs once per address, when a user picks one: suggest matches
as they type, resolve the pick, fetch the open map data around it, decide
whether it is a home the generator can rebuild, and freeze the result into a
:class:`~canopy.contracts.SiteSnapshot`. Generation itself never comes here;
it reads the snapshot offline (see :mod:`canopy.worldgen.snapshot`).

``tests/test_architecture.py`` holds the network boundary: no module outside
this package may import an HTTP or socket library.
"""

from __future__ import annotations
