#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC example producing EM modes.

import dill
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from matplotlib import colors
from openpmd_viewer import OpenPMDTimeSeries
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j1, jn, jn_zeros

from pywarpx import picmi

constants = picmi.constants

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

diag_dir = "diags/field_diags"

ts = OpenPMDTimeSeries(diag_dir, check_all_files=True)


def transform_spatially(data_for_transform):
    # interpolate from regular r-grid to special r-grid
    interp = RegularGridInterpolator(
        (info.z, info.r), data_for_transform, method="linear"
    )
    data_interp = interp((zg, rg))

    # Applying manual hankel in r
    # Fmz = np.sum(proj*data_for_transform, axis=(2,3))
    Fmz = np.einsum("ijkl,kl->ij", proj, data_interp)
    # Standard fourier in z
    Fmn = fft.fftshift(fft.fft(Fmz, axis=1), axes=1)
    return Fmn


def process(it):
    print(f"Processing iteration {it}", flush=True)
    field, info = ts.get_field("E", "y", iteration=it)
    F_k = transform_spatially(field)
    return F_k


# grab the first iteration to get the grids
Bz, info = ts.get_field("B", "z", iteration=0)

nr = len(info.r)
nz = len(info.z)

nkr = 12  # number of radial modes to solve for

r_max = np.max(info.r)

# create r-grid with points spaced out according to zeros of the Bessel function
r_grid = jn_zeros(1, nr) / jn_zeros(1, nr)[-1] * r_max

zg, rg = np.meshgrid(info.z, r_grid)

# Setup Hankel Transform
j_1M = jn_zeros(1, nr)[-1]
r_modes = np.arange(nkr)

A = (
    4.0
    * np.pi
    * r_max**2
    / j_1M**2
    * j1(np.outer(jn_zeros(1, max(r_modes) + 1)[r_modes], jn_zeros(1, nr)) / j_1M)
    / jn(2, jn_zeros(1, nr)) ** 2
)

# No transformation for z
B = np.identity(nz)

# combine projection arrays
proj = np.einsum("ab,cd->acbd", A, B)

results = np.zeros((len(ts.t), nkr, nz), dtype=complex)
for ii, it in enumerate(ts.iterations):
    results[ii] = process(it)

# now Fourier transform in time
F_kw = fft.fftshift(fft.fft(results, axis=0), axes=0)

dz = info.z[1] - info.z[0]
kz = 2 * np.pi * fft.fftshift(fft.fftfreq(F_kw[0].shape[1], dz))
dt = ts.iterations[1] - ts.iterations[0]
omega = 2 * np.pi * fft.fftshift(fft.fftfreq(F_kw.shape[0], sim.dt * dt))

# Save data for future plotting purposes
np.savez("diags/spectrograms.npz", F_kw=F_kw, dz=dz, kz=kz, dt=dt, omega=omega)

# plot the resulting dispersions
k = np.linspace(0, 250, 500)
kappa = k * sim.l_i

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.75, 5))

vmin = [2e-3, 1.5e-3, 7.5e-4, 5e-4]
vmax = 1.0

# plot m = 1
for ii, m in enumerate([1, 3, 6, 8]):
    ax = axes.flatten()[ii]
    ax.set_title(f"m = {m}", fontsize=11)
    m -= 1
    pm1 = ax.pcolormesh(
        kz * sim.l_i,
        omega / sim.w_ci,
        abs(F_kw[:, m, :]) / np.max(abs(F_kw[:, m, :])),
        norm=colors.LogNorm(vmin=vmin[ii], vmax=vmax),
        cmap="inferno",
    )
    cb = fig.colorbar(pm1, ax=ax)
    cb.set_label(r"Normalized $E_\theta(k_z, m, \omega)$")

    # Get dispersion relation - see for example
    # T. Stix, Waves in Plasmas (American Inst. of Physics, 1992), Chap 6, Sec 2
    nu_m = jn_zeros(1, m + 1)[-1] / sim.Lr
    R2 = 0.5 * (nu_m**2 * (1.0 + kappa**2) + k**2 * (kappa**2 + 2.0))
    P4 = k**2 * (nu_m**2 + k**2)
    omega_fast = sim.vA * np.sqrt(R2 + np.sqrt(R2**2 - P4))
    omega_slow = sim.vA * np.sqrt(R2 - np.sqrt(R2**2 - P4))
    # Upper right corner
    ax.plot(k * sim.l_i, omega_fast / sim.w_ci, "w--", label="$\omega_{fast}$")
    ax.plot(
        k * sim.l_i,
        omega_slow / sim.w_ci,
        color="white",
        linestyle="--",
        label="$\omega_{slow}$",
    )
    # Thermal resonance
    thermal_res = sim.w_ci + 3 * sim.v_ti * k
    ax.plot(
        k * sim.l_i,
        thermal_res / sim.w_ci,
        color="magenta",
        linestyle="--",
        label="$\omega = \Omega_i + 3v_{th,i}k$",
    )
    ax.plot(
        -k * sim.l_i, thermal_res / sim.w_ci, color="magenta", linestyle="--", label=""
    )
    thermal_res = sim.w_ci - 3 * sim.v_ti * k
    ax.plot(
        k * sim.l_i,
        thermal_res / sim.w_ci,
        color="magenta",
        linestyle="--",
        label="$\omega = \Omega_i + 3v_{th,i}k$",
    )
    ax.plot(
        -k * sim.l_i, thermal_res / sim.w_ci, color="magenta", linestyle="--", label=""
    )


for ax in axes.flatten():
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(0, 1.6)

axes[0, 0].set_ylabel("$\omega/\Omega_{ci}$")
axes[1, 0].set_ylabel("$\omega/\Omega_{ci}$")
axes[1, 0].set_xlabel("$k_zl_i$")
axes[1, 1].set_xlabel("$k_zl_i$")

plt.savefig("normal_modes_disp.png", dpi=600)
if not sim.test:
    plt.show()
else:
    plt.close()

    # check if power spectrum sampling match earlier results
    amps = np.abs(F_kw[2, 1, len(kz) // 2 - 2 : len(kz) // 2 + 2])
    print("Amplitude sample: ", amps)
    assert np.allclose(
        amps, np.array([61.02377286, 19.80026021, 100.47687017, 10.83331295])
    )
