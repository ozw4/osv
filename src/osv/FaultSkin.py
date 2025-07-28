from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt
from typing import Iterable, Iterator, List, Optional, Sequence

from .FaultCell import ColorMap, FaultCell


@dataclass
class TriangleGroup:
    lhc: bool
    xyz: List[float]
    rgb: List[float]


@dataclass
class QuadGroup:
    lhc: bool
    xyz: List[float]
    rgb: List[float]


class FaultSkin:
    """A linked list of fault cells used to analyze faults."""

    def __init__(self) -> None:
        self._seed: Optional[FaultCell] = None
        self._cell_list: List[FaultCell] = []
        self._cells_ab: Optional[List[List[FaultCell]]] = None
        self._cells_lr: Optional[List[List[FaultCell]]] = None

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def get_seed(self) -> Optional[FaultCell]:
        return self._seed

    def size(self) -> int:
        return len(self._cell_list)

    def get_cells(self) -> List[FaultCell]:
        return list(self._cell_list)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_cells_all(skins: Sequence["FaultSkin"]) -> List[FaultCell]:
        cells: List[FaultCell] = []
        for skin in skins:
            cells.extend(skin._cell_list)
        return cells

    @staticmethod
    def get_fl(n1: int, n2: int, n3: int, skins: Sequence["FaultSkin"]) -> List[List[List[float]]]:
        fl = [[[0.0 for _ in range(n1)] for _ in range(n2)] for _ in range(n3)]
        for skin in skins:
            for cell in skin:
                fl[cell.i3][cell.i2][cell.i1] = cell.fl
        return fl

    @staticmethod
    def count_cells(skins: Sequence["FaultSkin"]) -> int:
        return sum(skin.size() for skin in skins)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[FaultCell]:
        return iter(self._cell_list)

    # ------------------------------------------------------------------
    # Cell arrays
    # ------------------------------------------------------------------
    def get_cells_ab(self) -> List[List[FaultCell]]:
        if self._cells_ab is not None:
            return self._cells_ab
        cell_set = set()
        cells_list: List[List[FaultCell]] = []
        for cell in self._cell_list:
            if cell not in cell_set:
                c = cell
                while c.ca is not None:
                    c = c.ca
                clist: List[FaultCell] = []
                while c is not None:
                    clist.append(c)
                    cell_set.add(c)
                    c = c.cb
                cells_list.append(clist)
        self._cells_ab = cells_list
        return self._cells_ab

    def get_cells_lr(self) -> List[List[FaultCell]]:
        if self._cells_lr is not None:
            return self._cells_lr
        cell_set = set()
        cells_list: List[List[FaultCell]] = []
        for cell in self._cell_list:
            if cell not in cell_set:
                c = cell
                while c.cl is not None and c.cl is not cell:
                    c = c.cl
                left = c
                clist = [c]
                cell_set.add(c)
                c = c.cr
                while c is not None and c is not left:
                    clist.append(c)
                    cell_set.add(c)
                    c = c.cr
                cells_list.append(clist)
        self._cells_lr = cells_list
        return self._cells_lr

    # ------------------------------------------------------------------
    # Processing utilities
    # ------------------------------------------------------------------
    def smooth_cell_normals(self, nsmooth: int) -> None:
        for _ in range(nsmooth):
            self._smooth_n(
                lambda c: [c.w1, c.w2, c.w3],
                lambda c, w: c.set_normal_vector(*w),
            )

    def get_cell_nearest_centroid(self) -> Optional[FaultCell]:
        c1 = c2 = c3 = cs = 0.0
        for cell in self._cell_list:
            c1 += cell.fl * cell.x1
            c2 += cell.fl * cell.x2
            c3 += cell.fl * cell.x3
            cs += cell.fl
        if cs == 0.0:
            return None
        c1 /= cs
        c2 /= cs
        c3 /= cs
        dmin = float("inf")
        cmin = None
        for cell in self._cell_list:
            d = cell.distance_squared_to(c1, c2, c3)
            if d < dmin:
                dmin = d
                cmin = cell
        return cmin

    def get_cell_xyz_uvw_rgb_for_likelihood(
        self, size: float, cmap: ColorMap, lhc: bool
    ) -> List[List[float]]:
        return FaultCell.get_xyz_uvw_rgb_for_likelihood(size, cmap, self.get_cells(), lhc)

    def get_cell_xyz_uvw_rgb_for_throw(
        self, size: float, cmap: ColorMap, lhc: bool
    ) -> List[List[float]]:
        return FaultCell.get_xyz_uvw_rgb_for_throw(size, cmap, self.get_cells(), lhc)

    def get_tri_mesh(self, cmap: ColorMap) -> TriangleGroup:
        rgb: List[float] = []
        xyz: List[float] = []
        for cell in self._cell_list:
            cr = cell.cr
            cb = cell.cb
            rb = cr.cb if cr else None
            br = cb.cr if cb else None
            if cr and rb:
                xyz.extend([cell.x3, cell.x2, cell.x1])
                rgb.append(cell.fl)
                xyz.extend([rb.x3, rb.x2, rb.x1])
                rgb.append(rb.fl)
                xyz.extend([cr.x3, cr.x2, cr.x1])
                rgb.append(cr.fl)
            if cb and br:
                xyz.extend([cell.x3, cell.x2, cell.x1])
                rgb.append(cell.fl)
                xyz.extend([cb.x3, cb.x2, cb.x1])
                rgb.append(cb.fl)
                xyz.extend([br.x3, br.x2, br.x1])
                rgb.append(br.fl)
        return TriangleGroup(True, xyz, cmap.get_rgb_floats(rgb))

    def get_quad_mesh_strike(self, cmap: ColorMap) -> QuadGroup:
        rgb: List[float] = []
        xyz: List[float] = []
        for cell in self._cell_list:
            cb = cell.cb
            cr = cell.cr
            br = cb.cr if cb else None
            fp = cell.fp
            if cr and br and cr:
                xyz.extend([cell.x3, cell.x2, cell.x1])
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([cb.x3, cb.x2, cb.x1])
                fp = cb.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([br.x3, br.x2, br.x1])
                fp = br.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([cr.x3, cr.x2, cr.x1])
                fp = cr.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
        return QuadGroup(True, xyz, cmap.get_rgb_floats(rgb))

    def get_tri_mesh_strike(self, cmap: ColorMap) -> TriangleGroup:
        rgb: List[float] = []
        xyz: List[float] = []
        for cell in self._cell_list:
            cr = cell.cr
            cb = cell.cb
            rb = cr.cb if cr else None
            br = cb.cr if cb else None
            fp = cell.fp
            if cr and rb:
                xyz.extend([cell.x3, cell.x2, cell.x1])
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([rb.x3, rb.x2, rb.x1])
                fp = rb.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([cr.x3, cr.x2, cr.x1])
                fp = cr.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
            if cb and br:
                xyz.extend([cell.x3, cell.x2, cell.x1])
                fp = cell.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([cb.x3, cb.x2, cb.x1])
                fp = cb.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
                xyz.extend([br.x3, br.x2, br.x1])
                fp = br.fp
                rgb.append(fp if fp <= 180 else 360 - fp)
        return TriangleGroup(True, xyz, cmap.get_rgb_floats(rgb))

    def update_strike(self) -> None:
        for cell in self._cell_list:
            cl = cell.cl
            cr = cell.cr
            fpl = fpr = scs = 0.0
            if cl is not None:
                d2 = cell.x2 - cl.x2
                d3 = cell.x3 - cl.x3
                ds = sqrt(d2 * d2 + d3 * d3)
                if ds != 0:
                    fpl = acos(d3 / ds)
                    scs += 1.0
            if cr is not None:
                d2 = -cell.x2 + cr.x2
                d3 = -cell.x3 + cr.x3
                ds = sqrt(d2 * d2 + d3 * d3)
                if ds != 0:
                    fpr = acos(d3 / ds)
                    scs += 1.0
            if scs > 0.0:
                cell.fp = (fpl + fpr) / scs

    def get_x1max(self) -> float:
        return max((c.x1 for c in self._cell_list), default=0.0)

    def get_x2max(self) -> float:
        return max((c.x2 for c in self._cell_list), default=0.0)

    def get_x3max(self) -> float:
        return max((c.x3 for c in self._cell_list), default=0.0)

    def get_cell_links_xyz(self) -> List[List[float]]:
        cells_ab = self.get_cells_ab()
        cells_lr = self.get_cells_lr()
        ns_ab = len(cells_ab)
        ns_lr = len(cells_lr)
        xyz_all: List[List[float]] = []
        for iseg in range(ns_ab + ns_lr):
            cells = cells_ab[iseg] if iseg < ns_ab else cells_lr[iseg - ns_ab]
            ncell = len(cells)
            np = ncell
            if iseg >= ns_ab and cells[0].cl == cells[-1]:
                np += 1
            xyzi: List[float] = []
            for ip in range(np):
                cell = cells[ip % ncell]
                xyzi.extend([cell.x3, cell.x2, cell.x1])
            xyz_all.append(xyzi)
        return xyz_all

    # ------------------------------------------------------------------
    # I/O utilities
    # ------------------------------------------------------------------
    @staticmethod
    def read_from_file(file_name: str) -> "FaultSkin":
        skin = FaultSkin()
        with open(file_name, "rb") as f:
            import struct

            ncell = struct.unpack(">i", f.read(4))[0]
            cell_list: List[FaultCell] = []
            for _ in range(ncell):
                x1, x2, x3, fl, fp, ft = struct.unpack(
                    ">ffffff", f.read(24)
                )
                cell = FaultCell(x1, x2, x3, fl, fp, ft)
                cell.skin = skin
                cell_list.append(cell)
                cell.s1, cell.s2, cell.s3 = struct.unpack(">fff", f.read(12))
            for cell in cell_list:
                ida, idb, idl, idr = struct.unpack(">iiii", f.read(16))
                if ida != -2147483648:
                    cell.ca = cell_list[ida]
                if idb != -2147483648:
                    cell.cb = cell_list[idb]
                if idl != -2147483648:
                    cell.cl = cell_list[idl]
                if idr != -2147483648:
                    cell.cr = cell_list[idr]
        skin._cell_list = cell_list
        return skin

    @staticmethod
    def write_to_file(file_name: str, skin: "FaultSkin") -> None:
        with open(file_name, "wb") as f:
            import struct

            f.write(struct.pack(">i", skin.size()))
            for cell in skin:
                f.write(
                    struct.pack(
                        ">ffffff", cell.x1, cell.x2, cell.x3, cell.fl, cell.fp, cell.ft
                    )
                )
                f.write(struct.pack(">fff", cell.s1, cell.s2, cell.s3))
            for cell in skin:
                for nabor in (cell.ca, cell.cb, cell.cl, cell.cr):
                    if nabor is not None:
                        f.write(struct.pack(">i", nabor.id))
                    else:
                        f.write(struct.pack(">i", -2147483648))

    @staticmethod
    def read_from_file_slow(file_name: str) -> "FaultSkin":
        import pickle

        with open(file_name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def write_to_file_slow(file_name: str, skin: "FaultSkin") -> None:
        import pickle

        with open(file_name, "wb") as f:
            pickle.dump(skin, f)

    # ------------------------------------------------------------------
    # Package/internal methods
    # ------------------------------------------------------------------
    def add(self, cell: FaultCell) -> None:
        assert cell.skin is None
        cell.skin = self
        if self._seed is None:
            self._seed = cell
        self._cell_list.append(cell)
        self._cells_ab = None
        self._cells_lr = None

    def _smooth1(self, getter, setter) -> None:
        ncell = self.size()
        vals = [0.0] * ncell
        cnts = [0.0] * ncell
        cell_nabors: List[Optional[FaultCell]] = [None] * 4
        for icell, cell in enumerate(self._cell_list):
            val_cell = getter(cell)
            cell_nabors[0] = cell.ca
            cell_nabors[1] = cell.cb
            cell_nabors[2] = cell.cl
            cell_nabors[3] = cell.cr
            for nabor in cell_nabors:
                if nabor is not None:
                    val_nabor = getter(nabor)
                    vals[icell] += val_cell + val_nabor
                    cnts[icell] += 2.0
        for icell, cell in enumerate(self._cell_list):
            cnti = cnts[icell]
            vali = vals[icell] / (cnti if cnti > 0.0 else 1.0)
            setter(cell, vali)

    def _smooth_n(self, getter, setter) -> None:
        ncell = self.size()
        nval = len(getter(self._seed)) if self._seed else 0
        vals = [[0.0] * nval for _ in range(ncell)]
        cnts = [0.0] * ncell
        cell_nabors: List[Optional[FaultCell]] = [None] * 4
        for icell, cell in enumerate(self._cell_list):
            vals_cell = getter(cell)
            cell_nabors[0] = cell.ca
            cell_nabors[1] = cell.cb
            cell_nabors[2] = cell.cl
            cell_nabors[3] = cell.cr
            for nabor in cell_nabors:
                if nabor is not None:
                    vals_nabor = getter(nabor)
                    for ival in range(nval):
                        vals[icell][ival] += vals_cell[ival] + vals_nabor[ival]
                    cnts[icell] += 2.0
        for icell, cell in enumerate(self._cell_list):
            cnti = cnts[icell]
            scl = 1.0 / (cnti if cnti > 0.0 else 1.0)
            setter(cell, [v * scl for v in vals[icell]])
