#!/usr/bin/env python3

# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Check the pressure tensor (P_<species>) and heat flux (Q_<species>) diagnostics.

1. The gridded P, Q and T must equal the central moments computed here directly from the
   particle data (nearest-grid-point cell, two passes: cell mean, then moments about it).
   In RZ, the velocities are expressed in the local (r, theta, z) basis.
2. trace(P) / (3 n e) must equal the scalar temperature T_<species> (in eV).
3. The domain-averaged P and Q must agree with the closed-form moments of the injected
   distribution (a mixture of drifting Gaussians in u = gamma*v) within the sampling error.
   For P, the expected value includes the finite-sample bias of the per-cell estimator,
   which normalizes by sum(w) rather than applying a Bessel correction.
"""

import sys

import numpy as np
import yt
from scipy.constants import c, e, m_e, m_p

yt.funcs.mylog.setLevel(50)

PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

# Injection sources: density [m^-3], mean u/c and thermal u/c (per axis of the basis in
# which the moments are written), macroparticles per cell
ub = np.array([0.2, -0.1, 0.15])
e_beam = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
CASES = {
    "3d": dict(
        dirs="xyz",
        species={
            "electron": dict(
                mass=m_e,
                sources=[
                    (0.75e22, ub, np.full(3, 0.01), 64),
                    (0.25e22, ub + 0.05 * e_beam, np.full(3, 0.02), 64),
                ],
            ),
            "ion": dict(
                mass=m_p,
                sources=[
                    (
                        1.0e22,
                        np.array([0.01, 0.0, -0.02]),
                        np.array([3e-4, 4e-4, 5e-4]),
                        64,
                    )
                ],
            ),
        },
    ),
    # (r, theta, z) basis: rigid rotation u_theta = 0.1 c, bulk drift and beam along z
    "rz": dict(
        dirs="rtz",
        species={
            "electron": dict(
                mass=m_e,
                sources=[
                    (
                        0.75e22,
                        np.array([0.0, 0.1, 0.1]),
                        np.array([0.01, 0.01, 0.02]),
                        64,
                    ),
                    (
                        0.25e22,
                        np.array([0.0, 0.1, 0.15]),
                        np.array([0.01, 0.01, 0.02]),
                        64,
                    ),
                ],
            ),
        },
    ),
}


def closed_form(mass, sources):
    """P (3x3) and Q (3) of a Gaussian mixture, and the total density."""
    n = np.array([s[0] for s in sources])
    mu = np.array([s[1] for s in sources]) * c
    cov = np.array([np.diag(s[2] ** 2) for s in sources]) * c**2
    ntot = n.sum()
    frac = n / ntot
    mean = (frac[:, None] * mu).sum(axis=0)
    P = np.zeros((3, 3))
    Q = np.zeros(3)
    for f, m_k, c_k in zip(frac, mu, cov):
        d = m_k - mean
        P += f * (c_k + np.outer(d, d))
        Q += f * ((d @ d + np.trace(c_k)) * d + 2.0 * c_k @ d)
    return ntot * mass * P, 0.5 * ntot * mass * Q, ntot, cov


fn = sys.argv[1]
ds = yt.load(fn)
ad = ds.all_data()
geom = "rz" if any(f[1] == "particle_theta" for f in ds.field_list) else "3d"
case = CASES[geom]
dirs = case["dirs"]
P_NAMES = [f"P{dirs[i]}{dirs[j]}" for i, j in PAIRS]
Q_NAMES = [f"Q{d}" for d in dirs]

ndim = 2 if geom == "rz" else 3
ncell = tuple(int(n) for n in ds.domain_dimensions[:ndim])
lo = ds.domain_left_edge.v[:ndim]
dx = ((ds.domain_right_edge - ds.domain_left_edge) / ds.domain_dimensions).v[:ndim]
grid = ds.covering_grid(
    level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
)
if geom == "rz":
    r_lo = lo[0] + np.arange(ncell[0]) * dx[0]
    volume = np.repeat(
        (np.pi * ((r_lo + dx[0]) ** 2 - r_lo**2) * dx[1])[:, None], ncell[1], 1
    )
else:
    volume = np.full(ncell, np.prod(dx))
n_cells = np.prod(ncell)


def grid_field(name):
    return grid[("boxlib", name)].v.reshape(ncell)


def particle_moments(species, mass):
    """Two-pass NGP central moments, and sum(w^2)/sum(w), from the particle data."""
    pos = [ad[(species, f"particle_position_{a}")].v for a in "xyz"[:ndim]]
    u = np.stack([ad[(species, f"particle_momentum_{a}")].v / mass for a in "xyz"])
    if geom == "rz":
        theta = ad[(species, "particle_theta")].v
        cos, sin = np.cos(theta), np.sin(theta)
        u = np.stack([cos * u[0] + sin * u[1], -sin * u[0] + cos * u[1], u[2]])
    w = ad[(species, "particle_weight")].v
    idx = [
        np.clip(((p - p0) / h).astype(int), 0, n - 1)
        for p, p0, h, n in zip(pos, lo, dx, ncell)
    ]
    ic = np.ravel_multi_index(idx, ncell)
    nc = int(n_cells)
    V = volume.ravel()
    W = np.bincount(ic, w, nc)
    um = np.stack([np.bincount(ic, w * u[k], nc) for k in range(3)]) / W
    du = u - um[:, ic]
    du2 = (du**2).sum(axis=0)
    P = np.stack([np.bincount(ic, w * du[i] * du[j], nc) for i, j in PAIRS]) * mass / V
    Q = (
        np.stack([np.bincount(ic, w * du2 * du[k], nc) for k in range(3)])
        * 0.5
        * mass
        / V
    )
    T = np.bincount(ic, w * du2, nc) * mass / (3.0 * e) / W
    sw2_over_W = np.bincount(ic, w * w, nc) / W
    sh = tuple(ncell)
    return (
        P.reshape((6,) + sh),
        Q.reshape((3,) + sh),
        T.reshape(sh),
        (W / V).reshape(sh),
        (sw2_over_W / V).reshape(sh),
    )


# scipy and WarpX may use different CODATA releases for m_e/m_p, which enter the
# particle-side moments through the momentum-to-u conversion (relative change ~1e-10).
rtol_exact = 1e-9
nsigma = 5.0

for species, spec in case["species"].items():
    mass = spec["mass"]
    sources = spec["sources"]
    Pg = np.stack([grid_field(f"{name}_{species}") for name in P_NAMES])
    Qg = np.stack([grid_field(f"{name}_{species}") for name in Q_NAMES])
    Tg = grid_field(f"T_{species}")
    Pp, Qp, Tp, n, sw2_over_WV = particle_moments(species, mass)
    P_true, Q_true, ntot, cov = closed_form(mass, sources)

    # Finite-sample bias of the per-cell P estimator: E[P] = P_true - (m/V) sum_p w_p^2 C_p / W
    if geom == "rz":
        # weights vary within a cell; all sources share the same covariance here
        assert all(np.array_equal(cov[0], c_k) for c_k in cov)
        bias = mass * sw2_over_WV.mean() * cov[0]
    else:
        # equal weights n_k V / npc within each source
        V = volume.flat[0]
        bias = (
            sum(
                npc * (s[0] * V / npc) ** 2 * c_k
                for s, c_k, npc in zip(sources, cov, [s[3] for s in sources])
            )
            * mass
            / (ntot * V)
            / V
        )
    P_expected = np.array([(P_true - bias)[i, j] for i, j in PAIRS])
    P_inf = np.array([P_true[i, j] for i, j in PAIRS])

    P_scale = np.trace(P_true)
    Q_scale = max(np.abs(Q_true).max(), np.abs(Qg).max())

    # 1. Gridded moments == particle moments
    err_P = np.abs(Pg - Pp).max() / P_scale
    err_Q = np.abs(Qg - Qp).max() / Q_scale
    err_T = (np.abs(Tg - Tp) / Tp).max()
    print(f"{species}: grid vs particles: P {err_P:.2e}, Q {err_Q:.2e}, T {err_T:.2e}")
    assert err_P < rtol_exact and err_Q < rtol_exact and err_T < rtol_exact

    # 2. trace(P) / (3 n e) == T [eV]
    err_trace = (np.abs((Pg[0] + Pg[3] + Pg[5]) / (3.0 * n * e) - Tg) / Tg).max()
    print(f"{species}: trace(P)/(3 n e) vs T_{species}: {err_trace:.2e}")
    assert err_trace < rtol_exact

    # 3. Domain averages vs closed form
    for k, name in enumerate(P_NAMES):
        mean = Pg[k].mean()
        se = Pg[k].std(ddof=1) / np.sqrt(n_cells)
        print(
            f"{species}: {name} mean {mean:.5e}, expected {P_expected[k]:.5e} "
            f"(infinite-N {P_inf[k]:.5e}), {abs(mean - P_expected[k]) / se:.2f} sigma"
        )
        assert abs(mean - P_expected[k]) < nsigma * se + 1e-12 * P_scale
    for k, name in enumerate(Q_NAMES):
        mean = Qg[k].mean()
        se = Qg[k].std(ddof=1) / np.sqrt(n_cells)
        print(
            f"{species}: {name} mean {mean:.5e}, expected {Q_true[k]:.5e}, "
            f"{abs(mean - Q_true[k]) / se:.2f} sigma"
        )
        assert abs(mean - Q_true[k]) < nsigma * se
