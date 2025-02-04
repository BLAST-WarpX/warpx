#!/usr/bin/env python3

import numpy as np
from adios2 import Stream
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import hilbert

c = 3e8
lambda_laser = 800e-9
T_laser = lambda_laser / c  # utile ?

E_max = 3.26e10
T_peak = 10e-15
tau = 5e-15
delta_t = 1.6e-6 / c  # time for the laser to reach the particle

n_diags = 401
ex_t = np.zeros((n_diags,))
ey_t = np.zeros((n_diags,))
ez_t = np.zeros((n_diags,))
bx_t = np.zeros((n_diags,))
by_t = np.zeros((n_diags,))
bz_t = np.zeros((n_diags,))

def elec_d(x):
    return f"/data/particles/electron/{x}/__data__"

# Here we use adios2.Stream instead of openpmd_viewer.OpenPMDTimeSeries
# This is because the variable based encoding is still experimental
# and OpenPMDTimeSeries does not support it at the time of writing
# If there are only a few diagnostics, using file-based encoding will
# work, but opening/closing files creates A LOT of overhead and
# with a lot of saved iterations the reading time grows very fast
#
# This example should be updated in the future when OpenPMDTimeSeries
# supports variable-based encoding
with Stream("diags/diag1/openpmd.bp", "r") as s:
    for _ in s.steps():
        i = s.current_step()
        ex_t[i] = s.read(elec_d("ex")).item()
        ey_t[i] = s.read(elec_d("ey")).item()
        ez_t[i] = s.read(elec_d("ez")).item()
        bx_t[i] = s.read(elec_d("bx")).item()
        by_t[i] = s.read(elec_d("by")).item()
        bz_t[i] = s.read(elec_d("bz")).item()

DT = 6.324524234e-17  # @ CFL = 0.99
# on pourrait juste lire l'input file....
iterations = np.linspace(0, n_diags - 1, n_diags)
T = (iterations * DT) - DT / 2

T_peak_part = T_peak + delta_t
Ey_max = E_max * 1 / np.sqrt(5)  # polarisation vector (0 1 2)
Ez_max = E_max * np.sqrt(1 - 1 / 5)
By_max = Ez_max / c
Bz_max = Ey_max / c


def get_laser_th(E_0, T_p, T, tau, T_laser, phi=0):
    alpha = np.exp(-((T - T_peak_part) ** 2) / (tau**2))
    return E_0 * alpha * np.cos(2 * np.pi * (T - T_p) / T_laser + phi)


Ey_th = get_laser_th(Ey_max, T_peak_part, T, tau, T_laser)
Ez_th = get_laser_th(Ez_max, T_peak_part, T, tau, T_laser)
By_th = get_laser_th(By_max, T_peak_part, T, tau, T_laser, phi=np.pi)
Bz_th = get_laser_th(Bz_max, T_peak_part, T, tau, T_laser)

Ey_th = np.where(T >= delta_t, Ey_th, 0)
Ez_th = np.where(T >= delta_t, Ez_th, 0)
By_th = np.where(T >= delta_t, By_th, 0)
Bz_th = np.where(T >= delta_t, Bz_th, 0)


# due to injection method the simulated fields have a slight dephasing
# wrt the theoretical field
# here we compute the dephasing and correct it on the theoretical field
def align_cost(shift, sig1, sig2, time):
    # Create an interpolation function for the signal to be shifted
    interp_func = interp1d(time, sig1, kind="linear", fill_value="extrapolate")
    shifted_sig1 = interp_func(time + shift)
    # MSE
    cost = np.mean((shifted_sig1 - sig2) ** 2)
    return cost


init_shift = 0.0
res = minimize(
    align_cost,
    init_shift,
    args=(Ey_th, ey_t, T),
    method="Nelder-Mead",
    options={"xatol": 1e-20, "fatol": 1e-20, "maxiter": 10000},
)
opt_shift = res.x[0]

interp_func_ey = interp1d(T, Ey_th, kind="cubic", fill_value="extrapolate")
interp_func_ez = interp1d(T, Ez_th, kind="cubic", fill_value="extrapolate")
interp_func_by = interp1d(T, By_th, kind="cubic", fill_value="extrapolate")
interp_func_bz = interp1d(T, Bz_th, kind="cubic", fill_value="extrapolate")

Ey_aligned = interp_func_ey(T + opt_shift)
Ez_aligned = interp_func_ez(T + opt_shift)
By_aligned = interp_func_by(T + opt_shift)
Bz_aligned = interp_func_bz(T + opt_shift)

# verif

# E.1

assert np.allclose(np.zeros(ex_t.shape), ex_t, rtol=0, atol=5e-4)

# E.2

ey_p = ey_t[T >= delta_t]
Ey_p = Ey_aligned[T >= delta_t]
ey_p += 1e11
Ey_p += 1e11

ez_p = ez_t[T >= delta_t]
Ez_p = Ez_aligned[T >= delta_t]
ez_p += 1e11
Ez_p += 1e11

assert np.allclose(ey_p, Ey_p, rtol=8e-4, atol=0)
assert np.allclose(ez_p, Ez_p, rtol=2e-3, atol=0)

# E.3

ey_m = ey_t[T < delta_t - DT * 4]
ez_m = ez_t[T < delta_t - DT * 4]
assert np.allclose(ey_m, np.zeros(ey_m.shape), atol=1e-10, rtol=0)
assert np.allclose(ez_m, np.zeros(ey_m.shape), atol=1e-10, rtol=0)

# E.4

env_ez_t = np.abs(hilbert(ez_t))
env_ey_t = np.abs(hilbert(ey_t))
assert np.isclose(np.max(env_ez_t) / np.max(env_ey_t), 2, rtol=2e-3, atol=0)
assert np.allclose(2 * env_ey_t[85:390], env_ez_t[85:390], rtol=8e-3, atol=0)

# B.1

assert np.allclose(np.zeros(bx_t.shape), bx_t, rtol=0, atol=5e-4)

# B.2

by_p = by_t[T >= delta_t]
By_p = By_aligned[T >= delta_t]
by_p += 1e3
By_p += 1e3

bz_p = bz_t[T >= delta_t]
Bz_p = Bz_aligned[T >= delta_t]
bz_p += 1e3
Bz_p += 1e3

assert np.allclose(by_p, By_p, rtol=8e-4, atol=0)
assert np.allclose(bz_p, Bz_p, rtol=4e-4, atol=0)

# B.3

by_m = by_t[T < delta_t - DT * 4]
bz_m = bz_t[T < delta_t - DT * 4]
assert np.allclose(by_m, np.zeros(by_m.shape), atol=1e-10, rtol=0)
assert np.allclose(bz_m, np.zeros(by_m.shape), atol=1e-10, rtol=0)

# B.4

env_bz_t = np.abs(hilbert(bz_t))
env_by_t = np.abs(hilbert(by_t))
assert np.isclose(np.max(env_by_t) / np.max(env_bz_t), 2, rtol=2e-3, atol=0)
assert np.allclose(0.5 * env_by_t[85:390], env_bz_t[85:390], rtol=8e-3, atol=0)
