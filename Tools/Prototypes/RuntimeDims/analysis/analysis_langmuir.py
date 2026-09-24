#!/usr/bin/env python3

# Copyright 2026 Axel Huebl
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
#
# Validate a Langmuir-oscillation run of the runtime-dimensionality prototype
# (warpx.unified) against the analytic plasma-wave solution, for any
# dimensionality. The prototype stores fields in degenerate-3D plotfiles:
# collapsed dimensions have extent 1, the analytic contribution along them
# is 1.
#
# Usage: analysis_langmuir.py <plotfile>
#
# Cf. Examples/Tests/langmuir/analysis_{1,2,3}d.py; parameters must match
# Examples/Tests/langmuir/inputs_base_{1,2,3}d.
import sys

import numpy as np
import yt
from scipy.constants import c, e, epsilon_0, m_e

yt.funcs.mylog.setLevel(50)

fn = sys.argv[1]

# parameters of inputs_base_{1,2,3}d
epsilon = 0.01
n = 4.0e24  # 2e24 electrons + 2e24 positrons
n_osc = 2
wp = np.sqrt((n * e**2) / (m_e * epsilon_0))

ds = yt.load(fn)
t0 = ds.current_time.to_value()
dims = np.array(ds.domain_dimensions)
lo = np.array(ds.domain_left_edge)
hi = np.array(ds.domain_right_edge)
active = dims > 1
print(f"domain {dims}, active axes {active}, t = {t0}")

# wave vector along each active axis (collapsed axes do not contribute)
k = np.where(active, 2.0 * np.pi * n_osc / (hi - lo), 0.0)

data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)


def get_contribution(axis, is_cos):
    """analytic standing-wave factor along one axis, at cell centers"""
    if not active[axis]:
        return np.ones(1)
    du = (hi[axis] - lo[axis]) / dims[axis]
    u = lo[axis] + du * (0.5 + np.arange(dims[axis]))
    return np.cos(k[axis] * u) if is_cos else np.sin(k[axis] * u)


def get_theoretical_field(comp, t):
    """E_comp = epsilon m_e c^2 k_comp / q_e sin(k_comp u_comp) prod_cos sin(w_p t)"""
    amplitude = epsilon * (m_e * c**2 * k[comp]) / e * np.sin(wp * t)
    f = [get_contribution(ax, is_cos=(ax != comp)) for ax in range(3)]
    return (
        amplitude
        * f[0][:, np.newaxis, np.newaxis]
        * f[1][np.newaxis, :, np.newaxis]
        * f[2][np.newaxis, np.newaxis, :]
    )


error_rel = 0
for comp, field in enumerate(["Ex", "Ey", "Ez"]):
    if not active[comp]:
        continue
    E_sim = data[("boxlib", field)].to_ndarray()
    E_th = get_theoretical_field(comp, t0)
    max_error = abs(E_sim - E_th).max() / abs(E_th).max()
    print(f"{field}: max error: {max_error:.2e}")
    error_rel = max(error_rel, max_error)

tolerance_rel = 0.05
print(f"error_rel    : {error_rel}")
print(f"tolerance_rel: {tolerance_rel}")
assert error_rel < tolerance_rel
print("PASS")
