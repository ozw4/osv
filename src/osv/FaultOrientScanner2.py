from __future__ import annotations

import asyncio
from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin, sqrt
from typing import List, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter


@dataclass
class Sampling:
    """Uniform sampling description."""

    count: int
    delta: float
    first: float

    def get_count(self) -> int:
        return self.count

    def get_value(self, index: int) -> float:
        return self.first + self.delta * index


def _angle_sampling(sigma: float, amin: float, amax: float) -> Sampling:
    fa = amin
    da = np.degrees(0.5 / sigma)
    na = 1 + int((amax - amin) / da)
    da = (amax - amin) / (na - 1) if amax > amin else 1.0
    return Sampling(na, da, fa)


class SincInterpolator:
    """Simple bilinear interpolator with constant extrapolation."""

    @staticmethod
    def interpolate(
        n1: int,
        d1: float,
        f1: float,
        n2: int,
        d2: float,
        f2: float,
        array: Sequence[Sequence[float]],
        x1: float,
        x2: float,
    ) -> float:
        i1 = (x1 - f1) / d1
        i2 = (x2 - f2) / d2
        j1 = int(floor(i1))
        j2 = int(floor(i2))
        t1 = i1 - j1
        t2 = i2 - j2
        j1 = max(0, min(n1 - 2, j1))
        j2 = max(0, min(n2 - 2, j2))
        v00 = array[j2][j1]
        v10 = array[j2][j1 + 1]
        v01 = array[j2 + 1][j1]
        v11 = array[j2 + 1][j1 + 1]
        return (
            (1 - t1) * (1 - t2) * v00
            + t1 * (1 - t2) * v10
            + (1 - t1) * t2 * v01
            + t1 * t2 * v11
        )


class FaultOrientScanner2:
    """Scan for fault orientations."""

    def __init__(self, sigma1: float) -> None:
        self._sigma1 = float(sigma1)
        self._si = SincInterpolator()

    def get_theta_sampling(self, theta_min: float, theta_max: float) -> Sampling:
        return _angle_sampling(self._sigma1, theta_min, theta_max)

    def scan(
        self, theta_min: float, theta_max: float, g: Sequence[Sequence[float]]
    ) -> List[List[List[float]]]:
        st = self._make_theta_sampling(theta_min, theta_max)
        return self._scan_theta(st, g)

    def scan_dip(
        self, theta_min: float, theta_max: float, g: Sequence[Sequence[float]]
    ) -> List[List[List[float]]]:
        n2 = len(g)
        n1 = len(g[0])
        st1 = self._make_theta_sampling(90 - theta_max, 90 - theta_min)
        st2 = self._make_theta_sampling(90 + theta_min, 90 + theta_max)
        fp1 = self._scan_theta(st1, g)
        fp2 = self._scan_theta(st2, g)
        for i2 in range(n2):
            for i1 in range(n1):
                fx1 = fp1[0][i2][i1]
                fx2 = fp2[0][i2][i1]
                if fx1 < fx2:
                    fp1[0][i2][i1] = fx2
                    fp1[1][i2][i1] = fp2[1][i2][i1]
        return fp1

    def strive_convert(self, fp: Sequence[Sequence[float]]) -> List[List[float]]:
        n2 = len(fp)
        n1 = len(fp[0])
        fc = [[0.0] * n1 for _ in range(n2)]
        for i2 in range(n2):
            for i1 in range(n1):
                fpi = 90 - fp[i2][i1]
                if fpi < 0.0:
                    fpi += 180.0
                fc[i2][i1] = fpi
        return fc


    async def thin(self, fet: Sequence[List[List[float]]]) -> List[List[List[float]]]:
        n2 = len(fet[0])
        n1 = len(fet[0][0])
        f = fet[0]
        t = fet[1]
        ff = [[0.0] * n1 for _ in range(n2)]
        tt = [[0.0] * n1 for _ in range(n2)]
        pi = np.pi / 180.0

        async def process(i2: int) -> None:
            for i1 in range(n1):
                ti = t[i2][i1] * pi
                d1 = sin(ti)
                d2 = cos(ti)
                x1p = i1 + d1
                x2p = i2 + d2
                x1m = i1 - d1
                x2m = i2 - d2
                fi = f[i2][i1]
                fp = self._si.interpolate(n1, 1.0, 0.0, n2, 1.0, 0.0, f, x1p, x2p)
                fm = self._si.interpolate(n1, 1.0, 0.0, n2, 1.0, 0.0, f, x1m, x2m)
                if fp < fi and fm < fi:
                    ff[i2][i1] = fi
                    tt[i2][i1] = t[i2][i1]

        await asyncio.gather(*(process(i2) for i2 in range(n2)))
        return [ff, tt]


    def edge_like_fit2(
        self, r: int, fl: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        n2 = len(fl)
        n1 = len(fl[0])
        flr = [[0.0] * n1 for _ in range(n2)]
        for i2 in range(r, n2 - r - 1):
            for i1 in range(n1):
                ft = [fl[i2 + r2][i1] for r2 in range(-r, r + 1)]
                a, b, _ = self.parabolic_fit(ft)
                s = -abs(2 * a * r + b) / (2 * a)
                flr[i2][i1] = fl[i2][i1] / (s + 1.0e-4)
        return flr

    def edge_like_fit(
        self, r: int, el: Sequence[Sequence[float]], et: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        n2 = len(el)
        n1 = len(el[0])
        elr = [[0.0] * n1 for _ in range(n2)]
        pi = np.pi / 180.0
        si = SincInterpolator()
        for i2 in range(n2):
            for i1 in range(n1):
                ft = []
                ti = et[i2][i1] + 90.0
                if ti > 180.0:
                    ti -= 180.0
                ti *= pi
                t1 = cos(ti)
                t2 = sin(ti)
                for ir in range(-r, r + 1):
                    ft.append(
                        si.interpolate(
                            n1, 1.0, 0.0, n2, 1.0, 0.0, el, i1 + ir * t1, i2 + ir * t2
                        )
                    )
                a, b, _ = self.parabolic_fit(ft)
                s = -abs(2 * a * r + b) / (2 * a)
                elr[i2][i1] = el[i2][i1] / (s + 1.0e-4)
        return elr

    def edge_like_fit_g(
        self, r: int, el: Sequence[Sequence[float]], et: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        n2 = len(el)
        n1 = len(el[0])
        elr = [[0.0] * n1 for _ in range(n2)]
        pi = np.pi / 180.0
        si = SincInterpolator()
        hr = 5 * r
        for i2 in range(n2):
            for i1 in range(n1):
                ti = et[i2][i1] + 90.0
                if ti > 180.0:
                    ti -= 180.0
                ti *= pi
                t1 = cos(ti)
                t2 = sin(ti)
                ft = [
                    si.interpolate(
                        n1, 1.0, 0.0, n2, 1.0, 0.0, el, i1 + ir * t1, i2 + ir * t2
                    )
                    for ir in range(-hr, hr + 1)
                ]
                fs = gaussian_filter1d(ft, r, order=0, mode="nearest")
                f1 = gaussian_filter1d(ft, r, order=1, mode="nearest")
                f2 = gaussian_filter1d(ft, r, order=2, mode="nearest")
                elr[i2][i1] = fs[hr] * (-f2[hr] / (abs(f1[hr]) + 1.0e-4))
        return elr

    def parabolic_fit(self, f: Sequence[float]) -> List[float]:
        n1 = len(f)
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        for i1 in range(n1):
            x1 = float(i1)
            x2 = i1 * x1
            x3 = i1 * x2
            x4 = i1 * x3
            A[0, 0] += x4
            A[1, 0] += x3
            A[2, 0] += x2
            A[0, 1] += x3
            A[1, 1] += x2
            A[2, 1] += x1
            A[0, 2] += x2
            A[1, 2] += x1
            A[2, 2] += 1.0
            B[0, 0] += x2 * f[i1]
            B[1, 0] += x1 * f[i1]
            B[2, 0] += f[i1]
        x = np.linalg.solve(A, B)
        return [float(x[0, 0]), float(x[1, 0]), float(x[2, 0])]

    def _make_theta_sampling(self, theta_min: float, theta_max: float) -> Sampling:
        return _angle_sampling(self._sigma1, theta_min, theta_max)

    def _scan_theta(
        self, theta_sampling: Sampling, g: Sequence[Sequence[float]]
    ) -> List[List[List[float]]]:
        n2 = len(g)
        n1 = len(g[0])
        f = [[0.0] * n1 for _ in range(n2)]
        t = [[0.0] * n1 for _ in range(n2)]
        nt = theta_sampling.get_count()
        rsc = self.get_rotation_angle_map(n1, n2)
        ref2_sigma = 1.0
        ref1_sigma = self._sigma1
        for it in range(nt):
            ti = float(theta_sampling.get_value(it))
            theta = radians(ti)
            gr = self.rotate(theta, rsc, g)
            gr = gaussian_filter(gr, [0, ref2_sigma])
            gr = gaussian_filter(gr, [ref1_sigma, 0])
            s2 = self.unrotate(n1, n2, theta, rsc, gr)
            for i2 in range(n2):
                for i1 in range(n1):
                    st = s2[i2][i1]
                    st = 1.0 - st * st * st * st
                    if st > f[i2][i1]:
                        f[i2][i1] = st
                        t[i2][i1] = ti
        return [f, t]

    def rotate(
        self, theta: float, rsc: List[List[List[float]]], fx: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        n2 = len(fx)
        n1 = len(fx[0])
        nr = len(rsc[0])
        ri = (nr - 1) // 2
        h2 = int(floor(n2 / 2))
        h1 = int(floor(n1 / 2))
        r21 = abs(round(h2 * cos(theta) + h1 * sin(theta)))
        r22 = abs(round(h2 * cos(theta) - h1 * sin(theta)))
        r11 = abs(round(h1 * cos(theta) - h2 * sin(theta)))
        r12 = abs(round(h1 * cos(theta) + h2 * sin(theta)))
        r2 = max(r21, r22)
        r1 = max(r11, r12)
        m2 = r2 * 2 + 1
        m1 = r1 * 2 + 1
        st = sin(theta)
        ct = cos(theta)
        fr = [[0.0] * m1 for _ in range(m2)]

        async def process(i2: int) -> None:
            k2 = i2 + ri - r2
            rr2 = rsc[0][k2]
            ss2 = rsc[1][k2]
            cc2 = rsc[2][k2]
            for i1 in range(m1):
                k1 = i1 + ri - r1
                rk = rr2[k1]
                sk = ss2[k1]
                ck = cc2[k1]
                sp = sk * ct - ck * st
                cp = ck * ct + sk * st
                x1 = rk * cp + h1
                x2 = rk * sp + h2

                fr[i2][i1] = self._si.interpolate(
                    n1, 1.0, 0.0, n2, 1.0, 0.0, fx, x1, x2
                )

        asyncio.run(asyncio.gather(*(process(i2) for i2 in range(m2))))
        return fr

    def unrotate(
        self,
        n1: int,
        n2: int,
        theta: float,
        rsc: List[List[List[float]]],
        fr: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        h2 = int(floor(n2 / 2))
        h1 = int(floor(n1 / 2))
        nr = len(rsc[0])
        ri = (nr - 1) // 2
        m2 = len(fr)
        m1 = len(fr[0])
        r1 = (m1 - 1) // 2
        r2 = (m2 - 1) // 2
        st = sin(theta)
        ct = cos(theta)
        fu = [[0.0] * n1 for _ in range(n2)]

        async def process(i2: int) -> None:
            k2 = i2 + ri - h2
            rr2 = rsc[0][k2]
            ss2 = rsc[1][k2]
            cc2 = rsc[2][k2]
            for i1 in range(n1):
                k1 = i1 + ri - h1
                rk = rr2[k1]
                sk = ss2[k1]
                ck = cc2[k1]
                sp = sk * ct + ck * st
                cp = ck * ct - sk * st
                x1 = rk * cp + r1
                x2 = rk * sp + r2
                fu[i2][i1] = self._si.interpolate(
                    m1, 1.0, 0.0, m2, 1.0, 0.0, fr, x1, x2
                )

        asyncio.run(asyncio.gather(*(process(i2) for i2 in range(n2))))
        return fu

    def get_rotation_angle_map(self, n1: int, n2: int) -> List[List[List[float]]]:
        h1 = int(ceil(n1 / 2))
        h2 = int(ceil(n2 / 2))
        hr = round(sqrt(h1 * h1 + h2 * h2))
        nr = 2 * hr + 1
        sr = [[0.0] * nr for _ in range(nr)]
        cr = [[0.0] * nr for _ in range(nr)]
        rr = [[0.0] * nr for _ in range(nr)]
        for k2 in range(-hr, hr + 1):
            for k1 in range(-hr, hr + 1):
                i1 = k1 + hr
                i2 = k2 + hr
                ri = sqrt(k1 * k1 + k2 * k2)
                if ri == 0.0:
                    continue
                rc = 1.0 / ri
                rr[i2][i1] = ri
                sr[i2][i1] = k2 * rc
                cr[i2][i1] = k1 * rc
        return [rr, sr, cr]
