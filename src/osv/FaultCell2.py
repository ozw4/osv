"""FaultCell2 implementation in Python."""

from __future__ import annotations

from math import cos, radians, sin
from typing import List


class FaultCell2:
    """Represents a simple fault cell in 2D."""

    def __init__(self, i1: int, i2: int, fl: float, fp: float) -> None:
        """Initialize a FaultCell2.

        Parameters
        ----------
        i1, i2 : int
            Cell indices.
        fl : float
            Fault likelihood.
        fp : float
            Fault strike angle in degrees.
        """
        self._i1: int = i1
        self._i2: int = i2
        self._i3: int = 0
        self._fl: float = fl
        self._fp: float = fp
        self._ft: float = 0.0

    # ------------------------------------------------------------------
    # Basic getters
    # ------------------------------------------------------------------
    def get_i1(self) -> int:
        return self._i1

    def get_i2(self) -> int:
        return self._i2

    def get_index(self) -> List[int]:
        return [self._i1, self._i2, self._i3]

    def get_fl(self) -> float:
        return self._fl

    def get_fp(self) -> float:
        return self._fp

    def get_fault_normal(self) -> List[float]:
        return self.fault_normal_vector_from_strike(self._fp)

    def get_fault_strike_vector(self) -> List[float]:
        return self.fault_strike_vector_from_strike(self._fp)

    # ------------------------------------------------------------------
    # Static geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def fault_strike_vector_from_strike(phi: float) -> List[float]:
        p = radians(phi)
        cp = cos(p)
        sp = sin(p)
        v1 = -cp
        v2 = sp
        return [v1, v2]

    @staticmethod
    def fault_normal_vector_from_strike(phi: float) -> List[float]:
        p = radians(phi)
        cp = cos(p)
        sp = sin(p)
        u1 = sp
        u2 = cp
        return [u1, u2]

