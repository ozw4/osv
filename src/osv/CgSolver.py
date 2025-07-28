from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Optional

from .Vec import Vec

logger = getLogger(__name__)


class CgSolver:
    """Iterative conjugate-gradient solver for ``Ax=b``."""

    class Stop:
        TINY = "TINY"
        MAXI = "MAXI"
        USER = "USER"

    @dataclass
    class Info:
        stop: str
        niter: int
        bnorm: float
        rnorm: float

    class A:
        def apply(self, x: Vec, y: Vec) -> None:  # pragma: no cover - interface
            raise NotImplementedError

    class Stopper:
        def stop(self, info: "CgSolver.Info") -> bool:  # pragma: no cover
            raise NotImplementedError

    def __init__(self, tiny: float, maxi: int) -> None:
        self._tiny = tiny
        self._maxi = maxi

    def solve(
        self,
        stopper: Optional["CgSolver.Stopper"],
        a: "CgSolver.A",
        b: Vec,
        x: Vec,
    ) -> "CgSolver.Info":
        return self._solve(stopper, 0.0, a, b, x)

    def solve_precond(
        self,
        stopper: Optional["CgSolver.Stopper"],
        a: "CgSolver.A",
        m: "CgSolver.A",
        b: Vec,
        x: Vec,
    ) -> "CgSolver.Info":
        return self._solve_precond(stopper, 0.0, a, m, b, x)

    # ------------------------------------------------------------------
    # Internal CG implementations
    # ------------------------------------------------------------------
    def _solve(
        self,
        stopper: Optional["CgSolver.Stopper"],
        anorm: float,
        a: "CgSolver.A",
        b: Vec,
        x: Vec,
    ) -> "CgSolver.Info":
        q = b.clone()
        a.apply(x, q)
        r = b.clone()
        r.add(1.0, q, -1.0)
        d = r.clone()
        bnorm = b.norm2()
        rnorm = r.norm2()
        xnorm = x.norm2()
        logger.debug("begin: bnorm=%g rnorm=%g", bnorm, rnorm)
        info: Optional[CgSolver.Info] = None
        rrnorm = rnorm * rnorm
        iter_count = 0
        while (
            iter_count < self._maxi
            and rnorm > self._tiny * (anorm * xnorm + bnorm)
            and (info := self._user_stop(stopper, iter_count, bnorm, rnorm)) is None
        ):
            logger.debug("iter=%d rnorm=%g", iter_count, rnorm)
            a.apply(d, q)
            dq = d.dot(q)
            alpha = rrnorm / dq
            x.add(1.0, d, alpha)
            if anorm > 0.0:
                xnorm = x.norm2()
            if iter_count % 50 == 49:
                a.apply(x, q)
                r.add(0.0, b, 1.0)
                r.add(1.0, q, -1.0)
            else:
                r.add(1.0, q, -alpha)
            rrnorm_old = rrnorm
            rnorm = r.norm2()
            rrnorm = rnorm * rnorm
            beta = rrnorm / rrnorm_old
            d.add(beta, r, 1.0)
            iter_count += 1
        logger.debug("end: iter=%d rnorm=%g", iter_count, rnorm)
        if info is None:
            stop = self.Stop.TINY if iter_count < self._maxi else self.Stop.MAXI
            info = self.Info(stop, iter_count, bnorm, rnorm)
        return info

    def _solve_precond(
        self,
        stopper: Optional["CgSolver.Stopper"],
        anorm: float,
        a: "CgSolver.A",
        m: "CgSolver.A",
        b: Vec,
        x: Vec,
    ) -> "CgSolver.Info":
        q = b.clone()
        a.apply(x, q)
        r = b.clone()
        r.add(1.0, q, -1.0)
        s = r.clone()
        m.apply(r, s)
        d = s.clone()
        rsnorm = r.dot(s)
        bnorm = b.norm2()
        rnorm = r.norm2()
        xnorm = x.norm2()
        logger.debug("begin: bnorm=%g rnorm=%g", bnorm, rnorm)
        info: Optional[CgSolver.Info] = None
        iter_count = 0
        while (
            iter_count < self._maxi
            and rnorm > self._tiny * (anorm * xnorm + bnorm)
            and (info := self._user_stop(stopper, iter_count, bnorm, rnorm)) is None
        ):
            logger.debug("iter=%d rnorm=%g", iter_count, rnorm)
            a.apply(d, q)
            dq = d.dot(q)
            alpha = rsnorm / dq
            x.add(1.0, d, alpha)
            xnorm = x.norm2()
            if iter_count % 50 == 49:
                a.apply(x, q)
                r.add(0.0, b, 1.0)
                r.add(1.0, q, -1.0)
            else:
                r.add(1.0, q, -alpha)
            rnorm = r.norm2()
            m.apply(r, s)
            rsnorm_old = rsnorm
            rsnorm = r.dot(s)
            beta = rsnorm / rsnorm_old
            d.add(beta, s, 1.0)
            iter_count += 1
        logger.debug("end: iter=%d rnorm=%g", iter_count, rnorm)
        if info is None:
            stop = self.Stop.TINY if iter_count < self._maxi else self.Stop.MAXI
            info = self.Info(stop, iter_count, bnorm, rnorm)
        return info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _user_stop(
        self, stopper: Optional["CgSolver.Stopper"], iter_count: int, bnorm: float, rnorm: float
    ) -> Optional["CgSolver.Info"]:
        if stopper is not None:
            info = self.Info(self.Stop.USER, iter_count, bnorm, rnorm)
            if stopper.stop(info):
                return info
        return None
