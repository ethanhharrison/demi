#!/usr/bin/env python3
"""
Simple N-point moving average filter for streaming sensor data.

Works well on Raspberry Pi (no third-party deps). Designed for:
  - incremental/streaming use via MovingAverage.update(x)
  - one-shot/batch use via moving_average(seq, N)

Behavior:
  - During "warm-up" (fewer than N samples), the filter returns the
    average of the samples seen so far (common for real-time smoothing).
  - Set require_full_window=True to return None until the window fills.
"""

from collections import deque
from typing import Deque, Iterable, Iterator, List, Optional

class MovingAverage:
    def __init__(self, N: int, *, require_full_window: bool = False) -> None:
        if N <= 0:
            raise ValueError("N must be a positive integer")
        self.N: int = N
        self._q: Deque[float] = deque()
        self._sum: float = 0.0
        self.require_full_window: bool = require_full_window

    def reset(self) -> None:
        self._q.clear()
        self._sum = 0.0

    def update(self, x: float) -> Optional[float]:
        """
        Push a new sample and get the current moving-average.

        Returns:
          - float average
          - None if require_full_window=True and window not yet full
        """
        self._q.append(x)
        self._sum += x
        if len(self._q) > self.N:
            self._sum -= self._q.popleft()

        if self.require_full_window and len(self._q) < self.N:
            return None

        return self._sum / len(self._q)

def moving_average(seq: Iterable[float], N: int, *, require_full_window: bool = False) -> Iterator[Optional[float]]:
    """
    Batch helper: yields the moving average for each value in seq.
    """
    ma = MovingAverage(N, require_full_window=require_full_window)
    for v in seq:
        yield ma.update(float(v))

# Optional convenience for quick CLI testing:
if __name__ == "__main__":
    # Example: echo "1 2 3 4 5 6" | python3 filter.py 3
    import sys
    if len(sys.argv) < 2:
        print("Usage: filter.py <N>   # reads whitespace-separated numbers from stdin", file=sys.stderr)
        sys.exit(1)
    N = int(sys.argv[1])
    data = []
    for token in sys.stdin.read().split():
        try:
            data.append(float(token))
        except ValueError:
            pass
    for y in moving_average(data, N):
        print("" if y is None else f"{y:.6f}")

