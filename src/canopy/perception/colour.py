"""Colour arithmetic for classification: RGB to HSV, and HSV rules that respect noise.

Hue and saturation are what the detector's colour rules lean on, because the
sensor's shading multiplies all three channels of a surface's colour by one
factor. Brightness (value) changes with the angle to the sun, while hue and
saturation do not, so a meter in shadow and a meter in sunlight fall in the
same hue range.

Noise does not scale with the colour, though. A hue is an angle measured from
the gap between the brightest and dimmest channels (the chroma), so a pale or
shadowed surface, with only a few levels of chroma, has a hue that one noisy
sample barely constrains: a blue-grey conduit in shadow reads anywhere from
green to violet hit by hit, yet averages to within a few degrees of its true
hue over a few hundred hits. :meth:`ColourRule.admits` therefore widens each
bound by the colour's standard error when asked to, which rejects a sample
only when it is confidently outside the rule, and the error shrinks as more
hits are pooled into the colour being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ["ColourRule", "Hsv", "rgb_to_hsv"]

#: Degrees in a full turn of hue.
_FULL_TURN_DEG = 360.0
#: A hue slack this wide admits every hue, whatever the arc.
_ANY_HUE_DEG = 180.0
#: Degrees of hue per unit of (channel difference / chroma): the hexcone's sectors.
_DEG_PER_SECTOR = 60.0
#: Largest value of an 8-bit colour channel.
_CHANNEL_MAX = 255.0
#: A difference of two independent channels has sqrt(2) times one's noise.
_ROOT2 = float(np.sqrt(2.0))

FloatArray = npt.NDArray[np.float64]


class Hsv(NamedTuple):
    """Colours in hue, saturation and value, each shape ``(N,)``."""

    hue: FloatArray
    """Degrees in ``[0, 360)``; 0 for a grey, which has none."""
    sat: FloatArray
    """``chroma / val``, in ``[0, 1]``."""
    val: FloatArray
    """Brightest channel, in ``[0, 1]``."""
    chroma: FloatArray
    """Brightest minus dimmest channel, in ``[0, 1]``."""


def rgb_to_hsv(rgb: npt.NDArray[Any]) -> Hsv:
    """Convert colours on a 0-255 scale, shape ``(N, 3)``, to :class:`Hsv`.

    Any real dtype is accepted: pooled (averaged) colours are fractional.
    """
    c = np.asarray(rgb, dtype=np.float64) / _CHANNEL_MAX
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    top = c.max(axis=1)
    chroma = top - c.min(axis=1)
    sat = np.divide(chroma, top, out=np.zeros_like(top), where=top > 0.0)
    safe = np.where(chroma > 0.0, chroma, 1.0)
    sector = np.where(
        top == r,
        np.mod((g - b) / safe, 6.0),
        np.where(top == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(chroma > 0.0, sector * _DEG_PER_SECTOR, 0.0)
    return Hsv(hue=hue, sat=sat, val=top, chroma=chroma)


@dataclass(frozen=True, slots=True)
class ColourRule:
    """A box in HSV space that one class's colour falls in.

    ``hue_deg`` is an arc: ``lo <= hi`` runs from ``lo`` up to ``hi``, while
    ``lo > hi`` wraps through 360, which is how a red class (around 0) is
    written. A span of 360 admits every hue.
    """

    hue_deg: tuple[float, float]
    sat: tuple[float, float]
    val: tuple[float, float]

    def admits(
        self,
        hsv: Hsv,
        *,
        noise: float | FloatArray = 0.0,
        tolerance: float = 0.0,
    ) -> npt.NDArray[np.bool_]:
        """Whether each colour lies inside the rule, allowing for measurement noise.

        Parameters
        ----------
        hsv
            The colours tested.
        noise
            Standard deviation of each channel of each colour, on the ``[0, 1]``
            scale: a scalar, or one value per colour. For a mean of ``n`` hits
            it is the single-hit noise over ``sqrt(n)``.
        tolerance
            Standard errors of slack on every bound. Zero tests the rule
            exactly, whatever ``noise`` says.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)``.
        """
        slack = tolerance * np.broadcast_to(np.asarray(noise, dtype=np.float64), hsv.hue.shape)
        # Standard errors of hue (degrees) and saturation for this noise. A
        # noisy colour with no chroma, or no brightness, has no hue, or
        # saturation, to measure, so given any slack at all those bounds admit
        # it; tested exactly (no slack), it gets none.
        some_slack = slack > 0.0
        hue_slack = np.minimum(
            np.divide(
                _DEG_PER_SECTOR * _ROOT2 * slack,
                hsv.chroma,
                out=np.where(some_slack, _ANY_HUE_DEG, 0.0),
                where=hsv.chroma > 0.0,
            ),
            _ANY_HUE_DEG,
        )
        sat_slack = np.divide(
            _ROOT2 * slack,
            hsv.val,
            out=np.where(some_slack, 1.0, 0.0),
            where=hsv.val > 0.0,
        )

        lo, hi = self.hue_deg
        arc = hi - lo if lo <= hi else hi - lo + _FULL_TURN_DEG
        span = arc + 2.0 * hue_slack
        in_hue = (span >= _FULL_TURN_DEG) | (
            np.mod(hsv.hue - (lo - hue_slack), _FULL_TURN_DEG) <= span
        )
        in_sat = (hsv.sat >= self.sat[0] - sat_slack) & (hsv.sat <= self.sat[1] + sat_slack)
        in_val = (hsv.val >= self.val[0] - slack) & (hsv.val <= self.val[1] + slack)
        return np.asarray(in_hue & in_sat & in_val, dtype=np.bool_)
