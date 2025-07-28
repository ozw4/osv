from __future__ import annotations

from math import sqrt, ulp
from typing import List, Sequence

from .Vec import Vec


class VecArrayFloat2(Vec):
    """Vector represented by a 2-D ``float`` array."""

    def __init__(self, arg1: int | Sequence[Sequence[float]], n2: int | None = None) -> None:
        if isinstance(arg1, int):
            if n2 is None:
                raise ValueError("n2 must be provided when arg1 is int")
            self._n1 = arg1
            self._n2 = n2
            self._a = [[0.0 for _ in range(self._n1)] for _ in range(self._n2)]
        else:
            self._a = [list(row) for row in arg1]
            self._n2 = len(self._a)
            self._n1 = len(self._a[0]) if self._n2 > 0 else 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_array(self) -> List[List[float]]:
        return self._a

    def get_n1(self) -> int:
        return self._n1

    def get_n2(self) -> int:
        return self._n2

    # ------------------------------------------------------------------
    # Vec interface
    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        return ulp(1.0)

    def clone(self) -> "VecArrayFloat2":
        data = [row[:] for row in self._a]
        return VecArrayFloat2(data)

    def dot(self, vthat: Vec) -> float:
        athis = self._a
        athat = vthat.get_array()  # type: ignore[attr-defined]
        total = 0.0
        for i2 in range(self._n2):
            row_this = athis[i2]
            row_that = athat[i2]
            for i1 in range(self._n1):
                total += row_this[i1] * row_that[i1]
        return total

    def norm2(self) -> float:
        total = 0.0
        for row in self._a:
            for val in row:
                total += val * val
        return sqrt(total)

    def zero(self) -> None:
        for row in self._a:
            for i in range(self._n1):
                row[i] = 0.0

    def scale(self, s: float) -> None:
        for row in self._a:
            for i in range(self._n1):
                row[i] *= s

    def add(self, sthis: float, vthat: Vec, sthat: float) -> None:
        athis = self._a
        athat = vthat.get_array()  # type: ignore[attr-defined]
        for i2 in range(self._n2):
            row_this = athis[i2]
            row_that = athat[i2]
            for i1 in range(self._n1):
                row_this[i1] = row_this[i1] * sthis + row_that[i1] * sthat
