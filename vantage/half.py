"""One half of one match, in the single layout every later stage reads.

Everything downstream (noise simulation, features, the model) works on this structure, so the
open training matches and the competition match go through exactly the same code after loading.

Conventions
-----------
* Metres, origin at the centre spot, x along the length of the pitch, y across it.
* The home team attacks towards +x. Loaders flip a half that is played the other way, so
  "forward for home" always means "+x" and the two teams are mirror images of each other.
* 25 frames per second, one row per frame, no gaps.
* A missing observation is NaN. That holds for the ball and for every player slot.
* Players are anonymous: a column of ``home_xy`` is a track slot, not a person.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import A, D, H, X


@dataclass
class Half:
    match: str
    period: int                       # 1 or 2
    pitch: tuple                      # (length, width) in metres
    ball_xy: np.ndarray               # (T, 2) float32, NaN where the ball is not seen
    home_xy: np.ndarray               # (T, S_home, 2) float32, NaN where a slot is empty
    away_xy: np.ndarray               # (T, S_away, 2) float32
    labels: Optional[np.ndarray] = None   # (T,) int8 in {H, A, D, X}; None for unlabelled data
    ball_z: Optional[np.ndarray] = None   # (T,) float32 height in metres, when the source has it
    first_frame: int = 0              # frame index of row 0 in the source's own numbering
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return int(self.ball_xy.shape[0])

    def swapped(self):
        """The same half seen from the other side: teams exchanged, pitch rotated half a turn.

        After the swap the former away team is "home" and still attacks +x, so a model trained on
        both views cannot learn anything that depends on which team is called home.
        """
        labels = None
        if self.labels is not None:
            labels = self.labels.copy()
            labels[self.labels == H] = A
            labels[self.labels == A] = H
        meta = dict(self.meta)
        if "restarts" in meta:
            other = {H: A, A: H}
            meta["restarts"] = [(row, other.get(team)) for row, team in meta["restarts"]]
        return replace(
            self,
            ball_xy=-self.ball_xy,
            home_xy=-self.away_xy,
            away_xy=-self.home_xy,
            labels=labels,
            meta=dict(meta, swapped=not self.meta.get("swapped", False)),
        )

    def check(self):
        """Raise on a malformed half. Cheap, and run by every loader before returning."""
        t = self.n_frames
        if self.ball_xy.shape != (t, 2):
            raise ValueError("ball_xy shape {}".format(self.ball_xy.shape))
        for name in ("home_xy", "away_xy"):
            array = getattr(self, name)
            if array.ndim != 3 or array.shape[0] != t or array.shape[2] != 2:
                raise ValueError("{} shape {}".format(name, array.shape))
        if self.labels is not None:
            if self.labels.shape != (t,):
                raise ValueError("labels shape {}".format(self.labels.shape))
            if not np.isin(self.labels, (H, A, D, X)).all():
                raise ValueError("labels outside H/A/D/X")
        if self.ball_z is not None and self.ball_z.shape != (t,):
            raise ValueError("ball_z shape {}".format(self.ball_z.shape))
        return self


def home_attacks_positive_x(home_xy, away_xy):
    """True when the home team defends the -x goal, judged from where the two teams stand.

    Over a whole half each team's average position sits in its own half of the pitch, however
    much one side dominates, so the sign of the gap between the two averages is the direction of
    play. It needs no kick-off frame and no goalkeeper identity, which automatic tracking does
    not provide.
    """
    home_mean = np.nanmean(home_xy[..., 0])
    away_mean = np.nanmean(away_xy[..., 0])
    return bool(home_mean < away_mean)
