from __future__ import annotations

import asyncio
from math import exp, sqrt
from typing import List, Sequence

FLT_MAX = float("inf")


class OptimalPathPicker:
    """Pick optimal paths in 2-D/3-D arrays."""

    def __init__(self, gate: int, an: float) -> None:
        self._gate = gate
        self._an = an

    def apply_transform(self, fx: Sequence[Sequence[float]]) -> List[List[float]]:
        """Transpose array for internal use."""
        n2 = len(fx)
        n1 = len(fx[0])
        ft = [[0.0] * n2 for _ in range(n1)]
        for i2 in range(n2):
            for i1 in range(n1):
                ft[i1][i2] = fx[i2][i1]
        return ft

    def apply_for_weight(self, vel: Sequence[Sequence[float]]) -> List[List[float]]:
        """Weight array with exponential decay."""
        n2 = len(vel)
        n1 = len(vel[0])
        w = [[0.0] * n2 for _ in range(n1)]
        for i2 in range(n2):
            for i1 in range(n1):
                w[i1][i2] = exp(-vel[i2][i1])
        return w

    def apply_for_weight_x(self, vel: Sequence[Sequence[float]]) -> List[List[float]]:
        """Weight array with exponential decay (transposed)."""
        n2 = len(vel)
        n1 = len(vel[0])
        w = [[0.0] * n2 for _ in range(n1)]
        for i2 in range(n2):
            for i1 in range(n1):
                w[i2][i1] = exp(-vel[i2][i1])
        return w

    async def accumulate_inline(self, vel: List[List[List[float]]]) -> List[List[float]]:
        n3 = len(vel)
        n2 = len(vel[0])
        n1 = len(vel[0][0])
        p1 = [[0.0] * n3 for _ in range(n2)]

        async def process(i2: int) -> None:
            w1 = [[exp(-vel[i3][i2][i1]) for i1 in range(n1)] for i3 in range(n3)]
            tf = [[0.0] * n1 for _ in range(n3)]
            tb = [[0.0] * n1 for _ in range(n3)]
            p1[i2] = self.forward_pick(n1 - 1, w1, tf)
            i0 = round(p1[i2][n3 - 1])
            i0 = max(0, min(n1 - 1, i0))
            p1[i2] = self.backward_pick(i0, w1, tb)
            for i3 in range(n3):
                for i1 in range(n1):
                    vel[i3][i2][i1] = tf[i3][i1] + tb[i3][i1]

        await asyncio.gather(*(process(i2) for i2 in range(n2)))
        return p1

    async def accumulate_inline_i10(
        self, i10: int, vel: List[List[List[float]]]
    ) -> List[List[float]]:
        n3 = len(vel)
        n2 = len(vel[0])
        n1 = len(vel[0][0])
        p1 = [[0.0] * n3 for _ in range(n2)]

        async def process(i2: int) -> None:
            w1 = [[exp(-vel[i3][i2][i1]) for i1 in range(n1)] for i3 in range(n3)]
            tf = [[0.0] * n1 for _ in range(n3)]
            tb = [[0.0] * n1 for _ in range(n3)]
            p1[i2] = self.forward_pick(i10, w1, tf)
            i0 = round(p1[i2][n3 - 1])
            i0 = max(0, min(n1 - 1, i0))
            p1[i2] = self.backward_pick(i10, w1, tb)
            for i3 in range(n3):
                for i1 in range(n1):
                    vel[i3][i2][i1] = tf[i3][i1] + tb[i3][i1]

        await asyncio.gather(*(process(i2) for i2 in range(n2)))
        return p1

    async def accumulate_crossline(
        self, p: Sequence[float], vel: List[List[List[float]]]
    ) -> List[List[float]]:
        n3 = len(vel)
        n2 = len(vel[0])
        n1 = len(vel[0][0])
        p2 = [[0.0] * n2 for _ in range(n3)]

        async def process(i3: int) -> None:
            vel3 = vel[i3]
            w2 = [[exp(-vel3[i2][i1]) for i1 in range(n1)] for i2 in range(n2)]
            tf = [[0.0] * n1 for _ in range(n2)]
            tb = [[0.0] * n1 for _ in range(n2)]
            i0 = round(p[i3])
            i0 = max(0, min(n1 - 1, i0))
            p2[i3] = self.forward_pick(i0, w2, tf)
            i0 = round(p2[i3][n2 - 1])
            i0 = max(0, min(n1 - 1, i0))
            p2[i3] = self.backward_pick(i0, w2, tb)
            for i2 in range(n2):
                for i1 in range(n1):
                    vel3[i2][i1] = tf[i2][i1] + tb[i2][i1]

        await asyncio.gather(*(process(i3) for i3 in range(n3)))
        return p2

    def backward_pick(self, i0: int, wx: List[List[float]], tx: List[List[float]] | None = None) -> List[float]:
        n2 = len(wx)
        n1 = len(wx[0])
        p = [0.0] * n2
        tt = [[0.0] * n1 for _ in range(n2)]
        what = [[0.0] * n1 for _ in range(n2)]
        prev = [0.0] * n1
        nextv = [0.0] * n1
        dist = [sqrt(i1 * i1 + self._an * self._an) for i1 in range(n1)]

        for i1 in range(n1):
            wi = 0.5 * (wx[n2 - 1][i1] + wx[n2 - 1][i0])
            tt[n2 - 1][i1] = abs(i1 - i0) * wi

        for i1 in range(n1):
            wi = 0.5 * (wx[n2 - 2][i1] + wx[n2 - 1][i0])
            prev[i1] = dist[abs(i1 - i0)] * wi
            what[n2 - 2][i1] = i0
            tt[n2 - 2][i1] = prev[i1]

        prob = [0.0] * (self._gate * 2 - 1)
        for i2 in range(n2 - 3, -1, -1):
            for i1 in range(n1):
                wi = wx[i2][i1]
                ib = max(i1 - self._gate, -1)
                ie = min(i1 + self._gate, n1)
                c = FLT_MAX
                ic = -1
                for i in range(ib + 1, ie):
                    w2 = 0.5 * (wi + wx[i2 + 1][i])
                    d = dist[abs(i1 - i)] * w2 + prev[i]
                    it = i - ib - 1
                    if d < c:
                        c = d
                        ic = it
                    prob[it] = d
                vs = self.find_minimum(ic, ie - ib - 1, ib + 1, c, what[i2][i1], prob)
                nextv[i1] = vs[0]
                what[i2][i1] = vs[1]
            for i1 in range(n1):
                prev[i1] = nextv[i1]
                tt[i2][i1] = prev[i1]
        self.forward_track(p, nextv, what)
        if tx is not None:
            for i2 in range(n2):
                for i1 in range(n1):
                    tx[i1][i2] = tt[i2][i1]
        return p

    def forward_pick(self, i0: int, wx: List[List[float]], tx: List[List[float]] | None = None) -> List[float]:
        n2 = len(wx)
        n1 = len(wx[0])
        p = [0.0] * n2
        tt = [[0.0] * n1 for _ in range(n2)]
        what = [[0.0] * n1 for _ in range(n2)]
        prev = [0.0] * n1
        nextv = [0.0] * n1
        dist = [sqrt(i1 * i1 + self._an * self._an) for i1 in range(n1)]

        for i1 in range(n1):
            wi = 0.5 * (wx[0][i1] + wx[0][i0])
            tt[0][i1] = abs(i1 - i0) * wi

        for i1 in range(n1):
            wi = 0.5 * (wx[1][i1] + wx[0][i0])
            prev[i1] = dist[abs(i1 - i0)] * wi
            what[1][i1] = i0
            tt[1][i1] = prev[i1]

        prob = [0.0] * (self._gate * 2 - 1)
        for i2 in range(2, n2):
            for i1 in range(n1):
                wi = wx[i2][i1]
                ib = max(i1 - self._gate, -1)
                ie = min(i1 + self._gate, n1)
                c = FLT_MAX
                ic = -1
                for i in range(ib + 1, ie):
                    w2 = 0.5 * (wi + wx[i2 - 1][i])
                    d = dist[abs(i1 - i)] * w2 + prev[i]
                    it = i - ib - 1
                    if d < c:
                        c = d
                        ic = it
                    prob[it] = d
                vs = self.find_minimum(ic, ie - ib - 1, ib + 1, c, what[i2][i1], prob)
                nextv[i1] = vs[0]
                what[i2][i1] = vs[1]
            for i1 in range(n1):
                prev[i1] = nextv[i1]
                tt[i2][i1] = prev[i1]
        self.backward_track(p, nextv, what)
        if tx is not None:
            for i2 in range(n2):
                for i1 in range(n1):
                    tx[i1][i2] = tt[i2][i1]
        return p

    @staticmethod
    def find_minimum(
        ic: int, nc: int, jc: int, c: float, pick: float, prob: Sequence[float]
    ) -> List[float]:
        if ic == 0:
            ic += 1
            fm = c
            f0 = prob[ic]
            fp = prob[ic + 1]
        elif nc - 1 == ic:
            ic -= 1
            fm = prob[ic - 1]
            f0 = prob[ic]
            fp = c
        else:
            fm = prob[ic - 1]
            f0 = c
            fp = prob[ic + 1]

        ic += jc
        a = fm + fp - 2.0 * f0
        if a <= 0.0:
            if fm < f0 and fm < fp:
                pick = ic - 1
                return [fm, pick]
            if fp < f0 and fp < fm:
                pick = ic + 1
                return [fp, pick]
            pick = ic
            return [f0, pick]

        b = 0.5 * (fm - fp)
        a = b / a
        if a > 1.0:
            pick = ic + 1
            return [fp, pick]
        if a < -1.0:
            pick = ic - 1
            return [fm, pick]
        if f0 < 0.5 * b * a:
            pick = ic
            return [f0, pick]
        f0 -= 0.5 * b * a
        pick = ic + a
        return [f0, pick]

    @staticmethod
    def forward_track(path: List[float], nextv: Sequence[float], what: Sequence[Sequence[float]]) -> None:
        n1 = len(nextv)
        n2 = len(path)
        c = FLT_MAX
        fc = 0.0
        for i1 in range(n1):
            d = nextv[i1]
            if d < c:
                c = d
                fc = float(i1)
        for i2 in range(n2):
            path[i2] = fc
            fc = OptimalPathPicker.interpolate(fc, i2, what)

    @staticmethod
    def backward_track(path: List[float], nextv: Sequence[float], what: Sequence[Sequence[float]]) -> None:
        n1 = len(nextv)
        n2 = len(path)
        c = FLT_MAX
        fc = 0.0
        for i1 in range(n1):
            d = nextv[i1]
            if d < c:
                c = d
                fc = float(i1)
        for i2 in range(n2 - 1, -1, -1):
            path[i2] = fc
            fc = OptimalPathPicker.interpolate(fc, i2, what)

    @staticmethod
    def interpolate(fc: float, i2: int, what: Sequence[Sequence[float]]) -> float:
        n1 = len(what[0])
        ic = round(fc - 0.5)
        fc -= ic
        if ic >= n1 - 1:
            return what[i2][n1 - 1]
        if ic < 0:
            return what[i2][0]
        return what[i2][ic] * (1.0 - fc) + what[i2][ic + 1] * fc

    @staticmethod
    def make_rhs_weights_inline(
        p23: Sequence[Sequence[float]],
        vel: Sequence[Sequence[Sequence[float]]],
        b: List[List[float]],
        ws: List[List[float]],
    ) -> None:
        n3 = len(vel)
        n2 = len(vel[0])
        n1 = len(vel[0][0])
        for i3 in range(n3):
            for i2 in range(n2):
                k23 = round(p23[i2][i3])
                k23 = max(0, min(n1 - 1, k23))
                w23i = vel[i3][i2][k23]
                w23i *= w23i
                ws[i3][i2] = w23i
                b[i3][i2] = p23[i2][i3] * w23i

    @staticmethod
    def make_rhs_weights(
        p23: Sequence[Sequence[float]],
        p32: Sequence[Sequence[float]],
        vel: Sequence[Sequence[Sequence[float]]],
        b: List[List[float]],
        ws: List[List[float]],
    ) -> None:
        n3 = len(vel)
        n2 = len(vel[0])
        n1 = len(vel[0][0])
        for i3 in range(n3):
            for i2 in range(n2):
                k23 = round(p23[i2][i3])
                k32 = round(p32[i3][i2])
                k23 = max(0, min(n1 - 1, k23))
                k32 = max(0, min(n1 - 1, k32))
                w23i = vel[i3][i2][k23]
                w32i = vel[i3][i2][k32]
                w23i *= w23i
                w32i *= w32i
                ws[i3][i2] = w23i + w32i
                b[i3][i2] = p23[i2][i3] * w23i + p32[i3][i2] * w32i

    @staticmethod
    def make_rhs_weights_1d(
        p1: Sequence[float],
        p2: Sequence[float],
        vel: Sequence[Sequence[float]],
        b: List[float],
        ws: List[float],
    ) -> None:
        n2 = len(vel)
        n1 = len(vel[0])
        for i1 in range(n1):
            k12 = round(p1[i1])
            k22 = round(p2[i1])
            k12 = max(0, min(n2 - 1, k12))
            k22 = max(0, min(n2 - 1, k22))
            w1i = vel[k12][i1]
            w2i = vel[k22][i1]
            w1i *= w1i
            w2i *= w2i
            ws[i1] = w1i + w2i
            b[i1] = p1[i1] * w1i + p2[i1] * w2i
