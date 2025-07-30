from __future__ import annotations

from enum import Enum
from typing import List, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


class RecursiveGaussianFilterP:
    """Recursive Gaussian smoothing and derivative filter."""

    class Method(Enum):
        DERICHE = "DERICHE"
        VAN_VLIET = "VAN_VLIET"

    def __init__(self, sigma: float, method: "RecursiveGaussianFilterP.Method" = Method.DERICHE) -> None:
        if sigma < 1.0:
            raise ValueError("sigma>=1.0")
        self._sigma = float(sigma)
        self._method = method

    # ------------------------------------------------------------------
    # 1-D filters
    # ------------------------------------------------------------------
    def apply0(self, x: Sequence[float], y: List[float]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, mode="nearest")
        self._assign1d(y, out)

    def apply1(self, x: Sequence[float], y: List[float]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, mode="nearest")
        self._assign1d(y, out)

    def apply2(self, x: Sequence[float], y: List[float]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, mode="nearest")
        self._assign1d(y, out)

    # ------------------------------------------------------------------
    # 2-D filters
    # ------------------------------------------------------------------
    def apply0X(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, axis=1, mode="nearest")
        self._assign2d(y, out)

    def apply1X(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, axis=1, mode="nearest")
        self._assign2d(y, out)

    def apply2X(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, axis=1, mode="nearest")
        self._assign2d(y, out)

    def applyX0(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, axis=0, mode="nearest")
        self._assign2d(y, out)

    def applyX1(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, axis=0, mode="nearest")
        self._assign2d(y, out)

    def applyX2(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, axis=0, mode="nearest")
        self._assign2d(y, out)

    def apply00(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 0), mode="nearest")
        self._assign2d(y, out)

    def apply10(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(1, 0), mode="nearest")
        self._assign2d(y, out)

    def apply01(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 1), mode="nearest")
        self._assign2d(y, out)

    def apply11(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(1, 1), mode="nearest")
        self._assign2d(y, out)

    def apply20(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(2, 0), mode="nearest")
        self._assign2d(y, out)

    def apply02(self, x: Sequence[Sequence[float]], y: List[List[float]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 2), mode="nearest")
        self._assign2d(y, out)

    # ------------------------------------------------------------------
    # 3-D filters
    # ------------------------------------------------------------------
    def apply0XX(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, axis=2, mode="nearest")
        self._assign3d(y, out)

    def apply1XX(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, axis=2, mode="nearest")
        self._assign3d(y, out)

    def apply2XX(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, axis=2, mode="nearest")
        self._assign3d(y, out)

    def applyX0X(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, axis=1, mode="nearest")
        self._assign3d(y, out)

    def applyX1X(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, axis=1, mode="nearest")
        self._assign3d(y, out)

    def applyX2X(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, axis=1, mode="nearest")
        self._assign3d(y, out)

    def applyXX0(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=0, axis=0, mode="nearest")
        self._assign3d(y, out)

    def applyXX1(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=1, axis=0, mode="nearest")
        self._assign3d(y, out)

    def applyXX2(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter1d(np.asarray(x, dtype=float), self._sigma, order=2, axis=0, mode="nearest")
        self._assign3d(y, out)

    def apply000(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 0, 0), mode="nearest")
        self._assign3d(y, out)

    def apply100(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(1, 0, 0), mode="nearest")
        self._assign3d(y, out)

    def apply010(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 1, 0), mode="nearest")
        self._assign3d(y, out)

    def apply001(self, x: Sequence[Sequence[Sequence[float]]], y: List[List[List[float]]]) -> None:
        out = gaussian_filter(np.asarray(x, dtype=float), sigma=self._sigma, order=(0, 0, 1), mode="nearest")
        self._assign3d(y, out)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    @staticmethod
    def _assign1d(out: List[float], data: np.ndarray) -> None:
        vals = data.tolist()
        for i, v in enumerate(vals):
            out[i] = float(v)

    @staticmethod
    def _assign2d(out: List[List[float]], data: np.ndarray) -> None:
        vals = data.tolist()
        for i, row in enumerate(vals):
            out[i][:] = [float(v) for v in row]

    @staticmethod
    def _assign3d(out: List[List[List[float]]], data: np.ndarray) -> None:
        vals = data.tolist()
        for i, plane in enumerate(vals):
            for j, row in enumerate(plane):
                out[i][j][:] = [float(v) for v in row]
