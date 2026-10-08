"""Analytic 1D cylindrical (screw-pinch) pressure-balance equilibrium.

Static radial force balance:

    d/dr [ p(r) + (B_z^2 + B_theta^2)/(2 mu0) ] + B_theta^2/(mu0 r) = 0

with profile choices (u = (r/a)^2):

    n(r)       = n0 * (f_b + (1 - f_b) * exp(-u))    (isothermal, p = 2 n k T)
    B_theta(r) = Bt * (r/a) / (1 + u)                (Gold-Hoyle-like)
    B_z(r)     = closed form from the balance integral,
                 B_z^2 = Bz0^2 + 2 mu0 (p(0) - p(r)) - B_theta^2
                         - Bt^2 * (1 - 1/(1+u))
    E_r(r)     = -(1/(e n)) dp_e/dr                  (electron force balance)
               = (2 k T r / (e a^2)) (1-f_b) e^{-u} / (f_b + (1-f_b) e^{-u})

Every Cartesian component is a C-infinity function of (x, y); the
equilibrium is independent of z.
"""

import numpy as np
from scipy.constants import e as q_e
from scipy.constants import mu_0

# Equilibrium parameters
A = 0.25  # pinch radius scale (m)
N0 = 1.0e19  # peak density (m^-3)
FB = 0.05  # edge density floor fraction
T_EV = 100.0  # temperature per species (eV), isothermal
BZ0 = 0.10  # on-axis axial field (T)
BT = 0.05  # azimuthal field scale (T)

KT = T_EV * q_e  # J
P0 = 2.0 * N0 * KT  # on-axis pressure (electrons + ions)

# WarpX grid used by the test
PROB_LO = np.array([-0.8, -0.8, 0.0])
PROB_HI = np.array([0.8, 0.8, 0.2])
NCELL = np.array([64, 64, 8])
H = (PROB_HI - PROB_LO) / NCELL  # = 0.025 m, uniform
NPAD = 2  # file lattice padding (cells) beyond the WarpX domain

# Yee staggering (fraction of a cell, relative to the nodal lattice)
YEE_POSITION = {
    "Ex": (0.5, 0.0, 0.0),
    "Ey": (0.0, 0.5, 0.0),
    "Ez": (0.0, 0.0, 0.5),
    "Bx": (0.0, 0.5, 0.5),
    "By": (0.5, 0.0, 0.5),
    "Bz": (0.5, 0.5, 0.0),
}
NODAL_POSITION = {c: (0.0, 0.0, 0.0) for c in YEE_POSITION}


def _u(x, y):
    return (x * x + y * y) / A**2


def n_of_u(u):
    """Density (m^-3)."""
    return N0 * (FB + (1.0 - FB) * np.exp(-u))


def p_of_u(u):
    """Total pressure (Pa), p = 2 n k T."""
    return 2.0 * n_of_u(u) * KT


def btheta_over_r(u):
    """B_theta / r (T/m); smooth at the axis."""
    return (BT / A) / (1.0 + u)


def bz_of_u(u):
    """Axial field from the pressure-balance integral (T)."""
    btheta2 = BT**2 * u / (1.0 + u) ** 2
    bz2 = (
        BZ0**2
        + 2.0 * mu_0 * (P0 - p_of_u(u))
        - btheta2
        - BT**2 * (1.0 - 1.0 / (1.0 + u))
    )
    return np.sqrt(bz2)


def er_over_r(u):
    """E_r / r (V/m^2); smooth at the axis."""
    expu = np.exp(-u)
    return (2.0 * KT / (q_e * A**2)) * (1.0 - FB) * expu / (FB + (1.0 - FB) * expu)


FIELD_FUNCS = {
    "Ex": lambda x, y, z: x * er_over_r(_u(x, y)),
    "Ey": lambda x, y, z: y * er_over_r(_u(x, y)),
    "Ez": lambda x, y, z: np.zeros_like(x),
    "Bx": lambda x, y, z: -y * btheta_over_r(_u(x, y)),
    "By": lambda x, y, z: x * btheta_over_r(_u(x, y)),
    "Bz": lambda x, y, z: bz_of_u(_u(x, y)),
}


def check_pressure_balance(rmax=2.0, nr=200001, rtol=1.0e-6):
    """Verify the radial force balance residual on a fine 1D grid.

    Returns the max residual normalized by p0/a; raises if above rtol.
    """
    r = np.linspace(0.0, rmax, nr)
    u = (r / A) ** 2
    btheta = r * btheta_over_r(u)
    ptot = p_of_u(u) + (bz_of_u(u) ** 2 + btheta**2) / (2.0 * mu_0)
    residual = np.gradient(ptot, r, edge_order=2)
    residual[1:] += btheta[1:] ** 2 / (mu_0 * r[1:])
    resmax = np.max(np.abs(residual)) / (P0 / A)
    if resmax > rtol:
        raise RuntimeError(f"pressure balance residual {resmax:.3e} > {rtol:.1e}")
    return resmax
