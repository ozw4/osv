"""Python translation of FaultCellGrid."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from math import inf


# Assume FaultCell is defined in FaultCell.py in the same package
from .FaultCell import FaultCell


class FaultCellGrid:
    """Fault cells in a 3D sampling grid."""

    def __init__(
        self,
        arg1: Union[int, Sequence[FaultCell]],
        arg2: Optional[int] = None,
        arg3: Optional[int] = None,
    ) -> None:
        if isinstance(arg1, Sequence) and arg2 is None and arg3 is None:
            self._init_from_cells(arg1)
        elif isinstance(arg1, int) and arg2 is not None and arg3 is not None:
            self._init_dimensions(arg1, arg2, arg3)
        else:
            raise ValueError("Invalid constructor arguments")

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_n1(self) -> int:
        return self._n1

    def get_n2(self) -> int:
        return self._n2

    def get_n3(self) -> int:
        return self._n3

    def get_i1_min(self) -> int:
        return self._j1

    def get_i2_min(self) -> int:
        return self._j2

    def get_i3_min(self) -> int:
        return self._j3

    def get_i1_max(self) -> int:
        return self._j1 + self._n1 - 1

    def get_i2_max(self) -> int:
        return self._j2 + self._n2 - 1

    def get_i3_max(self) -> int:
        return self._j3 + self._n3 - 1

    # ------------------------------------------------------------------
    # Cell accessors
    # ------------------------------------------------------------------
    def get(self, i1: int, i2: int, i3: int) -> Optional[FaultCell]:
        i1 -= self._j1
        i2 -= self._j2
        i3 -= self._j3
        if 0 <= i1 < self._n1 and 0 <= i2 < self._n2 and 0 <= i3 < self._n3:
            return self._cells[i3][i2][i1]
        return None

    def set(self, obj: Union[FaultCell, Iterable[FaultCell]]) -> None:
        if isinstance(obj, FaultCell):
            i1 = obj.i1 - self._j1
            i2 = obj.i2 - self._j2
            i3 = obj.i3 - self._j3
            self._cells[i3][i2][i1] = obj
        else:
            for c in obj:
                self.set(c)

    def set_cells_in_box(self, cell: FaultCell, d1: int, d2: int, d3: int) -> None:
        c1, c2, c3 = cell.i1, cell.i2, cell.i3
        b1 = max(c1 - d1, 0)
        b2 = max(c2 - d2, 0)
        b3 = max(c3 - d3, 0)
        e1 = min(c1 + d1, self._n1 - 1)
        e2 = min(c2 + d2, self._n2 - 1)
        e3 = min(c3 + d3, self._n3 - 1)
        for i3 in range(b3, e3 + 1):
            for i2 in range(b2, e2 + 1):
                for i1 in range(b1, e1 + 1):
                    ci = self._cells[i3][i2][i1]
                    if ci is not None:
                        ci.used = True

    def find_cell_in_box(
        self,
        c1: int,
        c2: int,
        c3: int,
        d1: int,
        d2: int,
        d3: int,
    ) -> Optional[FaultCell]:
        b1 = max(c1 - d1, 0)
        b2 = max(c2 - d2, 0)
        b3 = max(c3 - d3, 0)
        e1 = min(c1 + d1, self._n1 - 1)
        e2 = min(c2 + d2, self._n2 - 1)
        e3 = min(c3 + d3, self._n3 - 1)
        dx = float("inf")
        cell: Optional[FaultCell] = None
        for i3 in range(b3, e3 + 1):
            for i2 in range(b2, e2 + 1):
                for i1 in range(b1, e1 + 1):
                    ci = self._cells[i3][i2][i1]
                    di = (c1 - i1) ** 2 + (c2 - i2) ** 2 + (c3 - i3) ** 2
                    if ci is not None and ci.skin is None and di < dx:
                        dx = di
                        cell = ci
        return cell

    def find_cells_in_box(
        self,
        x1: float,
        x2: float,
        x3: float,
        fp: float,
        d1: int,
        d2: int,
        d3: int,
    ) -> bool:
        c1 = round(x1)
        c2 = round(x2)
        c3 = round(x3)
        b1 = max(c1 - d1, 0)
        b2 = max(c2 - d2, 0)
        b3 = max(c3 - d3, 0)
        e1 = min(c1 + d1, self._n1 - 1)
        e2 = min(c2 + d2, self._n2 - 1)
        e3 = min(c3 + d3, self._n3 - 1)
        if fp > 180:
            fp = 360 - fp
        for i3 in range(b3, e3 + 1):
            for i2 in range(b2, e2 + 1):
                for i1 in range(b1, e1 + 1):
                    ci = self._cells[i3][i2][i1]
                    if ci is not None:
                        fpi = ci.fp
                        if fpi > 180:
                            fpi = 360 - fpi
                        dp = abs(fp - fpi)
                        if dp < 40:
                            return True
        return False

    def find_cells_in_box_cell(
        self, cell: FaultCell, d1: int, d2: int, d3: int
    ) -> bool:
        c1, c2, c3 = cell.i1, cell.i2, cell.i3
        fp = cell.fp
        b1 = max(c1 - d1, 0)
        b2 = max(c2 - d2, 0)
        b3 = max(c3 - d3, 0)
        e1 = min(c1 + d1, self._n1 - 1)
        e2 = min(c2 + d2, self._n2 - 1)
        e3 = min(c3 + d3, self._n3 - 1)
        for i3 in range(b3, e3 + 1):
            for i2 in range(b2, e2 + 1):
                for i1 in range(b1, e1 + 1):
                    ci = self._cells[i3][i2][i1]
                    if ci is not None:
                        fpi = ci.fp
                        dp = abs(fp - fpi)
                        dp = min(dp, 360 - dp)
                        if dp < 40:
                            return True
        return False

    def find_cell_above(self, cell: Optional[FaultCell]) -> Optional[FaultCell]:
        if cell is None:
            return None
        if cell.ca is not None:
            return cell.ca
        return self._find_cell_above_below(True, cell)

    def find_cell_below(self, cell: Optional[FaultCell]) -> Optional[FaultCell]:
        if cell is None:
            return None
        if cell.cb is not None:
            return cell.cb
        return self._find_cell_above_below(False, cell)

    def find_cell_left(self, cell: Optional[FaultCell]) -> Optional[FaultCell]:
        if cell is None:
            return None
        if cell.cl is not None:
            return cell.cl
        return self._find_cell_left_right(True, cell)

    def find_cell_right(self, cell: Optional[FaultCell]) -> Optional[FaultCell]:
        if cell is None:
            return None
        if cell.cr is not None:
            return cell.cr
        return self._find_cell_left_right(False, cell)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _init_from_cells(self, cells: Sequence[FaultCell]) -> None:
        i1min = min(c.i1 for c in cells)
        i2min = min(c.i2 for c in cells)
        i3min = min(c.i3 for c in cells)
        i1max = max(c.i1 for c in cells)
        i2max = max(c.i2 for c in cells)
        i3max = max(c.i3 for c in cells)
        self._j1 = i1min
        self._j2 = i2min
        self._j3 = i3min
        self._n1 = 1 + i1max - i1min
        self._n2 = 1 + i2max - i2min
        self._n3 = 1 + i3max - i3min
        self._cells = [
            [[None for _ in range(self._n1)] for _ in range(self._n2)]
            for _ in range(self._n3)
        ]
        for cell in cells:
            self.set(cell)

    def _init_dimensions(self, n1: int, n2: int, n3: int) -> None:
        self._j1 = self._j2 = self._j3 = 0
        self._n1 = n1
        self._n2 = n2
        self._n3 = n3
        self._cells = [
            [[None for _ in range(n1)] for _ in range(n2)] for _ in range(n3)
        ]

    def _find_cell_above_below(
        self, above: bool, cell: FaultCell
    ) -> Optional[FaultCell]:
        i1, i2, i3 = cell.i1, cell.i2, cell.i3
        x1, x2, x3 = cell.x1, cell.x2, cell.x3
        u1, u2, u3 = cell.u1, cell.u2, cell.u3
        k1 = 1
        if above:
            k1 = -k1
            u1, u2, u3 = -u1, -u2, -u3
        cmin: Optional[FaultCell] = None
        dmin = inf
        for k3 in range(-1, 2):
            for k2 in range(-1, 2):
                c = self.get(i1 + k1, i2 + k2, i3 + k3)
                if c is not None:
                    d1 = c.x1 - x1
                    d2 = c.x2 - x2
                    d3 = c.x3 - x3
                    du = d1 * u1 + d2 * u2 + d3 * u3
                    if du > 0.0:
                        d1 -= du * u1
                        d2 -= du * u2
                        d3 -= du * u3
                        d = d1 * d1 + d2 * d2 + d3 * d3
                        if d < dmin:
                            cmin = c
                            dmin = d
        return cmin


    def _find_cell_left_right(self, left: bool, cell: FaultCell) -> Optional[FaultCell]:

        i1, i2, i3 = cell.i1, cell.i2, cell.i3
        x1, x2, x3 = cell.x1, cell.x2, cell.x3
        v1, v2, v3 = cell.v1, cell.v2, cell.v3
        if left:
            v1, v2, v3 = -v1, -v2, -v3
        cmin: Optional[FaultCell] = None
        dmin = inf
        for k3 in range(-5, 6):
            for k2 in range(-5, 6):
                if k2 == 0 and k3 == 0:
                    continue
                c = self.get(i1, i2 + k2, i3 + k3)
                if c is not None:
                    d1 = c.x1 - x1
                    d2 = c.x2 - x2
                    d3 = c.x3 - x3
                    dv = d1 * v1 + d2 * v2 + d3 * v3
                    if dv > 0.0:
                        d1 -= dv * v1
                        d2 -= dv * v2
                        d3 -= dv * v3
                        d = d1 * d1 + d2 * d2 + d3 * d3
                        if d < dmin:
                            cmin = c
                            dmin = d
        return cmin
