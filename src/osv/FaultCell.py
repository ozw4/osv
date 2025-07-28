"""FaultCell implementation in Python."""

from __future__ import annotations


from math import cos, radians, sin, sqrt
from typing import Callable, Iterable, List, Optional, Sequence

from .FaultGeometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_from_normal_vector,
    fault_dip_from_normal_vector,
)

=======
from math import atan2, cos, radians, sin, asin, sqrt
from typing import Callable, Iterable, List, Optional, Sequence


class ColorMap:
    """Simple grayscale color map."""

    def get_rgb_floats(self, values: Sequence[float]) -> List[float]:
        rgb: List[float] = []
        for v in values:
            rgb.extend([v, v, v])
        return rgb


# -----------------------------------------------------------------------------
# Fault geometry utilities
# -----------------------------------------------------------------------------

def fault_dip_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    p = radians(phi)
    t = radians(theta)
    cp = cos(p)
    sp = sin(p)
    ct = cos(t)
    st = sin(t)
    return [st, ct * cp, -ct * sp]


def fault_strike_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    p = radians(phi)
    cp = cos(p)
    sp = sin(p)
    return [0.0, sp, cp]


def fault_normal_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    p = radians(phi)
    t = radians(theta)
    cp = cos(p)
    sp = sin(p)
    ct = cos(t)
    st = sin(t)
    return [-ct, st * cp, -st * sp]


def fault_strike_from_normal_vector(w1: float, w2: float, w3: float) -> float:
    return (atan2(-w3, w2) * 180.0 / 3.141592653589793) % 360.0


def fault_dip_from_normal_vector(w1: float, w2: float, w3: float) -> float:
    return asin(-w1) * 180.0 / 3.141592653589793


# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------

class FloatList:
    def __init__(self) -> None:
        self._data: List[float] = []

    @property
    def n(self) -> int:
        return len(self._data)

    def add(self, value: float) -> None:
        self._data.append(float(value))

    def add_list(self, values: Iterable[float]) -> None:
        for v in values:
            self.add(v)

    def trim(self) -> Optional[List[float]]:
        return self._data[:] if self._data else None


# -----------------------------------------------------------------------------
# FaultCell class
# -----------------------------------------------------------------------------

class FaultCell:
    """Represents an oriented point located on a fault."""

    def __init__(self, x1: float, x2: float, x3: float, fl: float, fp: float, ft: float) -> None:
        self.ca: Optional[FaultCell] = None
        self.cb: Optional[FaultCell] = None
        self.cl: Optional[FaultCell] = None
        self.cr: Optional[FaultCell] = None
        self.skin = None
        self.emp: Optional[List[float]] = None
        self.used = False
        self.id = 0
        self.set_(x1, x2, x3, fl, fp, ft)
        self.p2 = 0.0
        self.p3 = 0.0
        self.smp = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.t1 = 0.0
        self.t2 = 0.0
        self.t3 = 0.0
        self.nlr = 0
        self.nab = 0

    # ------------------------------------------------------------------
    # Basic getters/setters
    # ------------------------------------------------------------------
    def get_fl(self) -> float:
        return self.fl

    def get_fp(self) -> float:
        return self.fp

    def get_ft(self) -> float:
        return self.ft

    def set_fl(self, fl: float) -> None:
        self.fl = fl

    def get_x(self) -> List[float]:
        return [self.x1, self.x2, self.x3]

    def get_x1(self) -> float:
        return self.x1

    def get_x2(self) -> float:
        return self.x2

    def get_x3(self) -> float:
        return self.x3

    def get_w(self) -> List[float]:
        return [self.w1, self.w2, self.w3]

    def get_w1(self) -> float:
        return self.w1

    def get_w2(self) -> float:
        return self.w2

    def get_w3(self) -> float:
        return self.w3

    def get_s(self) -> List[float]:
        return [self.s1, self.s2, self.s3]

    def get_s1(self) -> float:
        return self.s1

    def get_s2(self) -> float:
        return self.s2

    def get_s3(self) -> float:
        return self.s3

    def get_u1(self) -> float:
        return self.u1

    def get_u2(self) -> float:
        return self.u2

    def get_u3(self) -> float:
        return self.u3

    def get_v1(self) -> float:
        return self.v1

    def get_v2(self) -> float:
        return self.v2

    def get_v3(self) -> float:
        return self.v3

    def get_vs(self) -> float:
        return self.vs

    def get_us(self) -> float:
        return self.us

    def get_m1(self) -> int:
        return self.i1

    def get_m2(self) -> int:
        return self.i2m

    def get_m3(self) -> int:
        return self.i3m

    def get_p1(self) -> int:
        return self.i1

    def get_p2(self) -> int:
        return self.i2p

    def get_p3(self) -> int:
        return self.i3p

    def get_i1(self) -> int:
        return self.i1

    def get_i2(self) -> int:
        return self.i2

    def get_i3(self) -> int:
        return self.i3

    def get_t1(self) -> float:
        return self.t1

    def get_t2(self) -> float:
        return self.t2

    def get_t3(self) -> float:
        return self.t3

    def set_t1(self, t1: float) -> None:
        self.t1 = t1

    def set_t2(self, t2: float) -> None:
        self.t2 = t2

    def set_t3(self, t3: float) -> None:
        self.t3 = t3

    def set_unfault_shifts(self, ts: Sequence[float]) -> None:
        self.t1, self.t2, self.t3 = ts[0], ts[1], ts[2]

    def get_fault_normal(self) -> List[float]:
        return fault_normal_vector_from_strike_and_dip(self.fp, self.ft)

    def get_fault_dip_vector(self) -> List[float]:
        return fault_dip_vector_from_strike_and_dip(self.fp, self.ft)

    def get_fault_strike_vector(self) -> List[float]:
        return fault_strike_vector_from_strike_and_dip(self.fp, self.ft)

    # ------------------------------------------------------------------
    # Fault curve and trace
    # ------------------------------------------------------------------
    def get_fault_curve_xyz(self) -> Optional[List[float]]:
        xyz = FloatList()
        p = [self.x1, self.x2, self.x3]
        cell: FaultCell = self
        j1 = cell.i1
        while j1 == cell.i1:
            xyz.add(p[2])
            xyz.add(p[1])
            xyz.add(p[0])
            cell = cell.walk_up_dip_from(p)
            j1 -= 1
        na = xyz.n // 3
        cell = self
        p = [self.x1, self.x2, self.x3]
        cell = cell.walk_down_dip_from(p)
        j1 = self.i1 + 1
        while j1 == cell.i1:
            xyz.add(p[2])
            xyz.add(p[1])
            xyz.add(p[0])
            cell = cell.walk_down_dip_from(p)
            j1 += 1
        xyzs = xyz.trim()
        if xyzs is None:
            return None
        left = xyzs[3 : 3 * na]
        triples = [left[i : i + 3] for i in range(0, len(left), 3)]
        triples.reverse()
        xyzs[3 : 3 * na] = [c for t in triples for c in t]
        return xyzs

    def get_fault_trace_xyz(self) -> Optional[List[float]]:
        xyz = FloatList()
        xyz.add(self.x3)
        xyz.add(self.x2)
        xyz.add(self.x1)
        c = self.cl
        while c is not None and c is not self:
            xyz.add(c.x3)
            xyz.add(c.x2)
            xyz.add(c.x1)
            c = c.cl
        nl = xyz.n // 3
        if c is not self:
            c = self.cr
            while c is not None:
                xyz.add(c.x3)
                xyz.add(c.x2)
                xyz.add(c.x1)
                c = c.cr
        xyzs = xyz.trim()
        if xyzs is None:
            return None
        left = xyzs[3 : 3 * nl]
        triples = [left[i : i + 3] for i in range(0, len(left), 3)]
        triples.reverse()
        xyzs[3 : 3 * nl] = [c for t in triples for c in t]
        return xyzs

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def set_normal_vector(self, w1: float, w2: float, w3: float) -> None:
        fp = fault_strike_from_normal_vector(w1, w2, w3)
        ft = fault_dip_from_normal_vector(w1, w2, w3)
        self.set_(self.x1, self.x2, self.x3, self.fl, fp, ft)

    def distance_to(self, p1: float, p2: float, p3: float) -> float:
        return sqrt(self.distance_squared_to(p1, p2, p3))

    def distance_squared_to(self, p1: float, p2: float, p3: float) -> float:
        d1 = p1 - self.x1
        d2 = p2 - self.x2
        d3 = p3 - self.x3
        return d1 * d1 + d2 * d2 + d3 * d3

    def distance_from_plane_to(self, p1: float, p2: float, p3: float) -> float:
        return self.w1 * (p1 - self.x1) + self.w2 * (p2 - self.x2) + self.w3 * (p3 - self.x3)

    def get_cell_above_nearest_to(self, p1: float, p2: float, p3: float) -> Optional["FaultCell"]:
        cla = self.cl.ca if self.cl else None
        cra = self.cr.ca if self.cr else None
        return self.nearest_cell(self.ca, cla, cra, p1, p2, p3)

    def get_cell_below_nearest_to(self, p1: float, p2: float, p3: float) -> Optional["FaultCell"]:
        clb = self.cl.cb if self.cl else None
        crb = self.cr.cb if self.cr else None
        return self.nearest_cell(self.cb, clb, crb, p1, p2, p3)

    def walk_up_dip_from(self, p: List[float]) -> "FaultCell":
        cell: FaultCell = self
        p1, p2, p3 = p
        assert abs(cell.distance_from_plane_to(p1, p2, p3)) < 0.01
        p1 -= 1.0
        p2 -= cell.us * cell.u2
        p3 -= cell.us * cell.u3
        ca = cell.get_cell_above_nearest_to(p1, p2, p3)
        if ca is not None:
            cell = ca
            us = cell.us
            ws = us * us * cell.distance_from_plane_to(p1, p2, p3)
            p2 -= ws * cell.w2
            p3 -= ws * cell.w3
        assert abs(cell.distance_from_plane_to(p1, p2, p3)) < 0.01
        p[0], p[1], p[2] = p1, p2, p3
        return cell

    def walk_down_dip_from(self, p: List[float]) -> "FaultCell":
        cell: FaultCell = self
        p1, p2, p3 = p
        assert abs(cell.distance_from_plane_to(p1, p2, p3)) < 0.01
        p1 += 1.0
        p2 += cell.us * cell.u2
        p3 += cell.us * cell.u3
        cb = cell.get_cell_below_nearest_to(p1, p2, p3)
        if cb is not None:
            cell = cb
            us = cell.us
            ws = us * us * cell.distance_from_plane_to(p1, p2, p3)
            p2 -= ws * cell.w2
            p3 -= ws * cell.w3
        assert abs(cell.distance_from_plane_to(p1, p2, p3)) < 0.01
        p[0], p[1], p[2] = p1, p2, p3
        return cell

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    @staticmethod
    def nearest_cell(c1: Optional["FaultCell"], c2: Optional["FaultCell"], c3: Optional["FaultCell"], p1: float, p2: float, p3: float) -> Optional["FaultCell"]:
        ds1 = FaultCell.distance_squared(c1, p1, p2, p3)
        ds2 = FaultCell.distance_squared(c2, p1, p2, p3)
        ds3 = FaultCell.distance_squared(c3, p1, p2, p3)
        dsm = min(ds1, ds2, ds3)
        if dsm == ds1:
            return c1
        if dsm == ds2:
            return c2
        return c3

    @staticmethod
    def distance_squared(c: Optional["FaultCell"], p1: float, p2: float, p3: float) -> float:
        return c.distance_squared_to(p1, p2, p3) if c is not None else float("inf")

    @staticmethod
    def rotate_point(cp: float, sp: float, ct: float, st: float, x: Sequence[float]) -> List[float]:
        x1, x2, x3 = x
        y1 = ct * x1 + st * x2
        y2 = -cp * st * x1 + cp * ct * x2 + sp * x3
        y3 = sp * st * x1 - sp * ct * x2 + cp * x3
        return [y1, y2, y3]

    @staticmethod
    def get_xyz_uvw_rgb(size: float, cmap: ColorMap, cells: Sequence["FaultCell"], get1: Callable[["FaultCell"], float], lhc: bool) -> List[List[float]]:
        xyz = FloatList()
        uvw = FloatList()
        fcl = FloatList()
        size *= 0.5
        qa = [0.0, -size, -size]
        qb = [0.0, size, -size]
        qc = [0.0, size, size]
        qd = [0.0, -size, size]
        if lhc:
            qb, qc = qc, qb
            qd, qa = qa, qd
        for cell in cells:
            x1, x2, x3 = cell.x1, cell.x2, cell.x3
            w1, w2, w3 = cell.w1, cell.w2, cell.w3
            fp = radians(cell.fp)
            ft = radians(cell.ft)
            cp = cos(fp)
            sp = sin(fp)
            ct = cos(ft)
            st = sin(ft)
            ra = FaultCell.rotate_point(cp, sp, ct, st, qa)
            rb = FaultCell.rotate_point(cp, sp, ct, st, qb)
            rc = FaultCell.rotate_point(cp, sp, ct, st, qc)
            rd = FaultCell.rotate_point(cp, sp, ct, st, qd)
            a1, a2, a3 = x1 + ra[0], x2 + ra[1], x3 + ra[2]
            b1, b2, b3 = x1 + rb[0], x2 + rb[1], x3 + rb[2]
            c1, c2, c3 = x1 + rc[0], x2 + rc[1], x3 + rc[2]
            d1, d2, d3 = x1 + rd[0], x2 + rd[1], x3 + rd[2]
            xyz.add_list([a3, a2, a1, b3, b2, b1, c3, c2, c1, d3, d2, d1])
            uvw.add_list([w3, w2, w1] * 4)
            fc = get1(cell)
            fcl.add_list([fc] * 4)
        fc_values = fcl.trim()
        rgb = cmap.get_rgb_floats(fc_values or [])
        return [xyz.trim(), uvw.trim(), rgb]

    @staticmethod
    def get_xyz_uvw_rgb_for_throw(size: float, cmap: ColorMap, cells: Sequence["FaultCell"], lhc: bool) -> List[List[float]]:
        return FaultCell.get_xyz_uvw_rgb(size, cmap, cells, lambda c: c.s1, lhc)

    @staticmethod
    def get_xyz_uvw_rgb_for_likelihood(size: float, cmap: ColorMap, cells: Sequence["FaultCell"], lhc: bool) -> List[List[float]]:
        return FaultCell.get_xyz_uvw_rgb(size, cmap, cells, lambda c: c.fl, lhc)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------
    def set_(self, x1: float, x2: float, x3: float, fl: float, fp: float, ft: float) -> None:
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.fl = fl
        self.fp = fp
        self.ft = ft
        self.i1 = round(x1)
        self.i2 = round(x2)
        self.i3 = round(x3)
        u = fault_dip_vector_from_strike_and_dip(fp, ft)
        v = fault_strike_vector_from_strike_and_dip(fp, ft)
        w = fault_normal_vector_from_strike_and_dip(fp, ft)
        self.u1, self.u2, self.u3 = u
        self.us = 1.0 / self.u1
        self.v1, self.v2, self.v3 = v
        vm = max(abs(self.v2), abs(self.v3))
        self.vs = 1.0 / vm if vm != 0 else 0.0
        self.w1, self.w2, self.w3 = w
        self.i2m = self.i2p = self.i2
        self.i3m = self.i3p = self.i3
        if x2 > self.i2:
            self.i2p += 1
        elif x2 < self.i2:
            self.i2m -= 1
        if x3 > self.i3:
            self.i3p += 1
        elif x3 < self.i3:
            self.i3m -= 1
        if (self.i2p - self.i2m) * self.w2 < 0.0:
            self.i2m, self.i2p = self.i2p, self.i2m
        if (self.i3p - self.i3m) * self.w3 < 0.0:
            self.i3m, self.i3p = self.i3p, self.i3m
