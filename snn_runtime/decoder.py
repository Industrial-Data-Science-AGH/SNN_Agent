"""The k-of-w decision rule, as a stream (task P2, point 1).

The alarm rule is not a runtime invention. It already exists in
``snn_pipeline/stream_eval.py:count_alarms``, which is the single source of the
FA/h numbers the project selects models by, and it works offline on a whole
spike train::

    times = np.where(s)[0]
    alarms, i = 0, 0
    while i + k - 1 < len(times):
        if times[i + k - 1] - times[i] < w:
            alarms += 1
            i = searchsorted(times, times[i + k - 1] + refrac)
        else:
            i += 1

A live runtime cannot call that: it never has the whole train. What it can do is
be provably the same function, computed incrementally, and that is what this
class is. ``tests/runtime/test_decoder.py`` asserts the equivalence against the
real ``count_alarms`` on random trains whenever ``snn_pipeline`` is checked out,
so the decision that fires on the Pi and the FA/h that selects the model cannot
drift apart without a test going red.

Why they are the same function. The offline scan fires at the smallest ``i``
with ``times[i+k-1] - times[i] < w``, anchoring the alarm at ``times[i+k-1]``.
This class fires at the smallest ``m`` with ``times[m] - times[m-k+1] < w``.
``m = i + k - 1`` is monotone in ``i``, so the two pick the same spike. After an
alarm, ``searchsorted`` drops every spike before ``t_alarm + refrac``; here that
is ``_blocked_until``, and a dropped spike is not remembered either, which is
the same as leaving it behind the scan cursor.

The rule counts frames, not microseconds: ``load_manifest`` has already refused
a package whose window is not a whole number of ``dt_us`` steps, so the
conversion is exact and happens once, here.
"""

from __future__ import annotations

from collections import deque


class KOfWDecoder:
    """Fires when k spikes fall inside w frames, then sleeps for the cooldown."""

    def __init__(self, *, k: int, window_frames: int, cooldown_frames: int) -> None:
        if k < 1:
            raise ValueError("k must be at least one spike")
        if window_frames < 1:
            raise ValueError("the window must be at least one frame")
        if cooldown_frames < 0:
            raise ValueError("the cooldown cannot be negative")
        self.k = k
        self.window_frames = window_frames
        self.cooldown_frames = cooldown_frames
        self._recent: deque[int] = deque(maxlen=k)
        self._frame = 0
        self._blocked_until = 0
        self.alarms = 0

    @classmethod
    def from_manifest(cls, decoder: dict, dt_us: int) -> KOfWDecoder:
        return cls(
            k=int(decoder["threshold"]),
            window_frames=decoder["window_us"] // dt_us,
            cooldown_frames=decoder["cooldown_us"] // dt_us,
        )

    def reset(self) -> None:
        self._recent.clear()
        self._frame = 0
        self._blocked_until = 0
        self.alarms = 0

    @property
    def frame(self) -> int:
        """Frames consumed since the last reset."""
        return self._frame

    @property
    def cooling_down(self) -> bool:
        return self._frame < self._blocked_until

    def flush(self) -> None:
        """Forget the spikes seen so far, keep the cooldown and the frame count.

        Called when the input stream has a hole in it. The spikes before a gap
        and the spikes after it did not happen within one window of each other
        in any meaningful sense, and stitching them together would invent an
        alarm out of missing data. The cooldown survives because it is about
        what was already reported downstream, not about what was observed.
        """
        self._recent.clear()

    def step(self, spiked: bool) -> bool:
        """Consume one frame of the decision neuron. True on the alarm frame."""
        now = self._frame
        self._frame += 1
        if not spiked or now < self._blocked_until:
            return False
        self._recent.append(now)
        if len(self._recent) < self.k or now - self._recent[0] >= self.window_frames:
            return False
        self._blocked_until = now + self.cooldown_frames
        self._recent.clear()
        self.alarms += 1
        return True

    def skip(self, frames: int) -> None:
        """Advance the frame clock over frames that were never observed."""
        if frames < 0:
            raise ValueError("cannot skip backwards")
        self._frame += frames
