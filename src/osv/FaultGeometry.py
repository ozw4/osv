"""Geometry utilities for fault calculations."""

from __future__ import annotations

from math import acos, asin, atan2, cos, radians, sin
from typing import List, Sequence


# -----------------------------------------------------------------------------
# Vector helpers
# -----------------------------------------------------------------------------

def fault_dip_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    """Return fault dip vector for specified strike and dip angles."""
    p = radians(phi)
    t = radians(theta)
    cp = cos(p)
    sp = sin(p)
    ct = cos(t)
    st = sin(t)
    return [st, ct * cp, -ct * sp]


def fault_strike_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    """Return fault strike vector for specified strike and dip angles."""
    p = radians(phi)
    cp = cos(p)
    sp = sin(p)
    return [0.0, sp, cp]


def fault_normal_vector_from_strike_and_dip(phi: float, theta: float) -> List[float]:
    """Return fault normal vector for specified strike and dip angles."""
    p = radians(phi)
    t = radians(theta)
    cp = cos(p)
    sp = sin(p)
    ct = cos(t)
    st = sin(t)
    return [-ct, st * cp, -st * sp]


def fault_strike_from_dip_vector(u1: float, u2: float, u3: float) -> float:
    """Return fault strike angle (degrees) from dip vector."""
    if u2 == 0.0 and u3 == 0.0:
        raise ValueError("dip vector is not vertical")
    return range360(atan2(-u3, u2) * 180.0 / 3.141592653589793)


def fault_strike_from_dip_vector_seq(u: Sequence[float]) -> float:
    return fault_strike_from_dip_vector(u[0], u[1], u[2])


def fault_dip_from_dip_vector(u1: float, u2: float, u3: float) -> float:
    """Return fault dip angle (degrees) from dip vector."""
    return asin(u1) * 180.0 / 3.141592653589793


def fault_dip_from_dip_vector_seq(u: Sequence[float]) -> float:
    return fault_dip_from_dip_vector(u[0], u[1], u[2])


def fault_strike_from_strike_vector(v1: float, v2: float, v3: float) -> float:
    """Return fault strike angle (degrees) from strike vector."""
    return range360(atan2(v2, v3) * 180.0 / 3.141592653589793)


def fault_strike_from_strike_vector_seq(v: Sequence[float]) -> float:
    return fault_strike_from_strike_vector(v[0], v[1], v[2])


def fault_strike_from_normal_vector(w1: float, w2: float, w3: float) -> float:
    """Return fault strike angle (degrees) from normal vector."""
    if w2 == 0.0 and w3 == 0.0:
        raise ValueError("normal vector is not vertical")
    return range360(atan2(-w3, w2) * 180.0 / 3.141592653589793)


def fault_strike_from_normal_vector_seq(w: Sequence[float]) -> float:
    return fault_strike_from_normal_vector(w[0], w[1], w[2])


def fault_dip_from_normal_vector(w1: float, w2: float, w3: float) -> float:
    """Return fault dip angle (degrees) from normal vector."""
    return acos(-w1) * 180.0 / 3.141592653589793


def fault_dip_from_normal_vector_seq(w: Sequence[float]) -> float:
    return fault_dip_from_normal_vector(w[0], w[1], w[2])


def cross_product(u: Sequence[float], v: Sequence[float]) -> List[float]:
    """Return cross product of two vectors."""
    u1, u2, u3 = u
    v1, v2, v3 = v
    return [u3 * v2 - u2 * v3, u1 * v3 - u3 * v1, u2 * v1 - u1 * v2]


def range360(phi: float) -> float:
    """Return angle in range [0, 360] degrees."""
    while phi < 0.0:
        phi += 360.0
    while phi >= 360.0:
        phi -= 360.0
    return phi


def range180(phi: float) -> float:
    """Return angle in range [-180, 180] degrees."""
    while phi < -180.0:
        phi += 360.0
    while phi > 180.0:
        phi -= 360.0
    return phi
