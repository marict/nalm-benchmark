from __future__ import annotations

from typing import List, Tuple

# 9 interpolation/extrapolation pairs
# Note: row 5 uses a two-interval extrapolation.
RANGE_PAIRS: List[Tuple[List[float], List[float] | List[List[float]]]] = [
    ([-20.0, -10.0], [-40.0, -20.0]),
    # ([-2.0, -1.0], [-6.0, -2.0]),
    # ([-1.2, -1.1], [-6.1, -1.2]),
    # ([-0.2, -0.1], [-2.0, -0.2]),
    # ([-2.0, 2.0], [[-6.0, -2.0], [2.0, 6.0]]),
    # ([0.1, 0.2], [0.2, 2.0]),
    # ([1.0, 2.0], [2.0, 6.0]),
    # ([1.1, 1.2], [1.2, 6.0]),
    # ([10.0, 20.0], [20.0, 40.0]),
]
