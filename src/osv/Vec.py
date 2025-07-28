from __future__ import annotations

from abc import ABC, abstractmethod


class Vec(ABC):
    """Abstract vector interface."""

    @abstractmethod
    def epsilon(self) -> float:
        """Machine epsilon for this vector."""

    @abstractmethod
    def clone(self) -> "Vec":
        """Return a deep copy of this vector."""

    @abstractmethod
    def dot(self, vthat: "Vec") -> float:
        """Return the dot product with ``vthat``."""

    @abstractmethod
    def norm2(self) -> float:
        """Return the L2 norm of this vector."""

    @abstractmethod
    def zero(self) -> None:
        """Zero all elements of this vector."""

    @abstractmethod
    def scale(self, s: float) -> None:
        """Scale this vector by ``s``."""

    @abstractmethod
    def add(self, sthis: float, vthat: "Vec", sthat: float) -> None:
        """Update this vector by ``this*sthis + vthat*sthat``."""
