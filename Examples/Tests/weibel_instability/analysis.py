#!/usr/bin/env python3

# Copyright 2023-2026 Juliette Pech, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Check the exponential-growth phase of the 2D Weibel instability tests (Darwin and explicit).

Two electron beams stream in opposite directions along y, out of the x-z simulation plane. The
instability produces magnetic structures in that plane: those varying along x produce B_z, those
varying along z produce B_x, and B_y stays at the noise level. B_x and B_z grow equally fast in
theory, but the grid is finer in x than in z and the seed mode varies along x, so B_z dominates.

Grown from noise alone, B_z would be a mix of many Fourier modes, each with its own growth rate,
and the growth of <B_z^2> would be a blurred average of them. The input therefore seeds a single
mode B_z ~ sin(k x) (see the base input file). We follow that Fourier mode: while it grows as
exp(gamma * t), ln(power) is a straight line with slope 2 * gamma. We fit that line and compare
gamma with Weibel's cold-beam rate gamma(k) (see cold_growth_rate). The beams drift much faster
than their thermal speed, so thermal corrections are negligible. The finite grid and particle
shape slow the growth slightly, so gamma(k) is an upper bound.
"""

import math
import sys
from pathlib import Path

sys.path.append("../../../Tools/Parser/")
import matplotlib
from input_file_parser import parse_input_file

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, epsilon_0, m_e
from scipy.constants import e as q_e

# Minimum number of snapshots in the fit window.
MIN_FIT_POINTS = 6
# The fit starts at this omega_p * t, after the start-up phase. At first, the seed excites both
# the growing and the decaying solution, so the slope of ln(power) only settles to 2 * gamma
# after gamma * t ~ 2. With the explicit solver, the seed also launches a light wave at the same
# k, which makes the mode power oscillate; this fades relative to the growing mode over time.
FIT_START_OMEGA_P_T = 250.0
# Minimum rise of ln(mode power) over the fit window (a rise of 1 is a factor e in power), so
# that the fit measures growth, not noise.
MIN_FIT_LN_BZ_2_RISE = 1.0
# Allowed relative deviation of the fitted growth rate above and below the cold-beam gamma(k).
GAMMA_RTOL_UPPER = 0.10
GAMMA_RTOL_LOWER = 0.30


def largest_contiguous_block(indices):
    """Return the longest run of consecutive values in the sorted array *indices*.

    For example, [2, 3, 7, 8, 9] gives [7, 8, 9].
    """
    assert indices.size > 0
    # Split wherever neighbouring values differ by more than 1.
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    blocks = np.split(indices, split_points)
    return max(blocks, key=len)


def linear_fit(x, y):
    """Fit y = slope * x + intercept and return slope, intercept, and R^2.

    R^2 is 1 if the points lie exactly on the line and near 0 if the line explains almost nothing.
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r_sq


def save_field_animation(
    iterations, t, get_field_frame, vmin, vmax, colorbar_label, title_prefix, filename
):
    """Save a GIF of a 2D field, one frame per snapshot.

    Parameters
    ----------
    iterations : array of int
        Step number of each snapshot.
    t : array of float
        Time of each snapshot [s].
    get_field_frame : function
        Maps a step number to (data, info), as returned by openPMD-viewer's ``get_field``.
    vmin, vmax : float
        Color-scale limits, shared by all frames so that they can be compared.
    colorbar_label : str
        Color-bar label.
    title_prefix : str
        Frame title; the step number and time are appended.
    filename : str
        Output GIF file name.
    """
    fig, ax = plt.subplots()
    data, info = get_field_frame(iterations[0])
    im = ax.imshow(
        data,
        origin="lower",
        extent=info.imshow_extent,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    title = ax.set_title(
        f"{title_prefix}, iteration {iterations[0]}, time {t[0]:.2e} s"
    )
    ax.set_xlabel(f"{info.axes[1]} (m)")
    ax.set_ylabel(f"{info.axes[0]} (m)")

    def update(i):
        iteration = iterations[i]
        data, info = get_field_frame(iteration)
        im.set_data(data)
        im.set_extent(info.imshow_extent)
        title.set_text(f"{title_prefix}, iteration {iteration}, time {t[i]:.2e} s")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(iterations), interval=100)
    ani.save(filename, writer="pillow")
    plt.close(fig)


def load_input_deck():
    """Read ``warpx_used_inputs`` (all parameters actually used, base file included) into a dict."""
    input_file = Path("warpx_used_inputs")
    assert input_file.exists(), "Could not find warpx_used_inputs"
    return parse_input_file(input_file)


input_dict = load_input_deck()

species_names = input_dict["particles.species_names"]
electron_species = ["electrons_1", "electrons_2"]
ion_species = "ions"
assert species_names == electron_species + [ion_species], (
    f"Expected two electron species and a neutralizing ion background, got {species_names}"
)

if "geometry.dims" in input_dict:
    assert input_dict["geometry.dims"][0] == "2"

# Electron beam parameters. Momenta are given as u = gamma * beta. The mean of u is the drift;
# its standard deviation is sqrt(theta), with theta = k_B T / (m_e c^2).
momentum_components = ("ux", "uy", "uz")
u_means = np.array(
    [
        [
            float(input_dict[f"{species}.{component}_mean"][0])
            for component in momentum_components
        ]
        for species in electron_species
    ]
)
beta_means = u_means / np.sqrt(1.0 + u_means**2)
# Drift speed of each beam along y.
betas = np.abs(beta_means[:, 1])
# Both beams have density n_beam, modulated by the seed mode.
for species in electron_species:
    assert input_dict[f"{species}.profile"][0] == "parse_density_function"
n_beam = float(input_dict["my_constants.n_beam"][0])
u_stds = np.array(
    [
        [
            float(input_dict[f"{species}.{component}_std"][0])
            for component in momentum_components
        ]
        for species in electron_species
    ]
)
# The thermal spread must be isotropic.
np.testing.assert_allclose(u_stds[:, 1], u_stds[:, 0])
np.testing.assert_allclose(u_stds[:, 2], u_stds[:, 0])
thetas = u_stds[:, 0] ** 2
drift_dirs = ["y" if beta_mean[1] > 0.0 else "-y" for beta_mean in beta_means]
charges = [input_dict[f"{species}.charge"][0] for species in electron_species]
masses = [input_dict[f"{species}.mass"][0] for species in electron_species]

assert set(drift_dirs) == {"y", "-y"}, (
    f"Expected counter-streaming y beams, got {drift_dirs}"
)
assert charges == ["-q_e", "-q_e"]
assert masses == ["m_e", "m_e"]
assert np.all(betas > 0.0)
assert n_beam > 0.0
assert np.all(thetas > 0.0)
# The beams must mirror each other (same speed and temperature), so that the unperturbed plasma
# carries no net current.
np.testing.assert_allclose(betas, betas[0])
np.testing.assert_allclose(thetas, thetas[0])

beta = float(np.mean(betas))
# Total electron density.
n_0 = 2.0 * n_beam

# The Darwin solver needs a neutral periodic plasma: immobile, uniform ions cancel the electron
# charge.
assert input_dict[f"{ion_species}.charge"][0] == "q_e"
assert input_dict[f"{ion_species}.do_not_push"][0] == "1"
assert input_dict[f"{ion_species}.momentum_distribution_type"][0] == "at_rest"
np.testing.assert_allclose(float(input_dict[f"{ion_species}.density"][0]), n_0)
# Electron plasma frequency (without relativistic corrections).
omega_p = math.sqrt(n_0 * q_e**2 / (m_e * epsilon_0))
skin_depth = c / omega_p
# Cold-beam growth rate at short wavelengths (k d >> 1); see cold_growth_rate.
gamma_theory = beta * omega_p

# Field solver name, for the plot title.
evolve_scheme = input_dict.get("algo.evolve_scheme", ["explicit"])[0]
solver_name = {"semi_implicit_darwin": "Darwin", "explicit": "explicit"}.get(
    evolve_scheme, evolve_scheme
)
print(f"Field solver: {solver_name}")

max_step = int(input_dict["max_step"][0]) if "max_step" in input_dict else None

print(f"Plasma frequency omega_p: {omega_p:.6e} 1/s")
print(f"Cold asymptotic growth rate beta * omega_p: {gamma_theory:.6e} 1/s")
print(f"v_drift / v_thermal: {beta / math.sqrt(float(np.mean(thetas))):.6g}")

# Seed mode parameters (see the base input file).
Lx = float(input_dict["my_constants.Lx"][0])
seed_mode = int(input_dict["my_constants.seed_mode"][0])
seed_amplitude = float(input_dict["my_constants.seed_amplitude"][0])
prob_lo_x = float(input_dict["geometry.prob_lo"][0])
prob_hi_x = float(input_dict["geometry.prob_hi"][0])
n_cell_x = int(input_dict["amr.n_cell"][0])
np.testing.assert_allclose(
    Lx, prob_hi_x - prob_lo_x, err_msg="Lx must match the domain length"
)
# The seed must be non-uniform and below the Nyquist mode.
assert 0 < seed_mode < n_cell_x // 2
assert 0.0 < seed_amplitude < 1.0
seed_k_d = 2.0 * np.pi * seed_mode / Lx * skin_depth
print(
    f"Seed mode: {seed_mode} wavelengths along x, k d = {seed_k_d:.4g}, "
    f"amplitude {seed_amplitude:.3g}"
)

ts = OpenPMDTimeSeries("diags/diag1")
iterations = np.asarray(ts.iterations)
t = np.asarray(ts.t)
assert iterations.size == t.size
assert t.size >= MIN_FIT_POINTS, (
    f"Need at least {MIN_FIT_POINTS} field diagnostics to fit growth"
)
assert np.all(np.diff(t) > 0.0)

print(f"Final diagnostic time: {t[-1]:.6e} s")
# Guard against output left over from a different (longer) run.
if max_step is not None:
    assert iterations[-1] <= max_step, (
        "The diagnostics do not appear to come from the parsed input file: "
        f"final iteration {iterations[-1]} exceeds max_step {max_step}"
    )

# Grid averages of B_x^2, B_y^2, B_z^2 per snapshot (proportional to the magnetic energy density).
Bx_2 = np.empty_like(t)
By_2 = np.empty_like(t)
Bz_2 = np.empty_like(t)
# Color scale of the B_z animation.
Bz_abs_max = 0.0
# Mean charge density of electrons_1 in the first snapshot, before filaments form, and the
# largest deviation from it; together they set the color scale of the rho animation.
rho_spatial_mean = None
rho_dev_max = 0.0
# Fourier power of B_z per snapshot, normalized so that its sum over all modes is <B_z^2>
# (Parseval). Shape: (snapshot, grid axis 0, grid axis 1).
Bz_mode_power = None
# All snapshots must share the grid shape; a mismatch means leftover output from another run.
grid_shape = None

for i, iteration in enumerate(iterations):
    Bx, _ = ts.get_field(field="B", coord="x", iteration=iteration, plot=False)
    By, _ = ts.get_field(field="B", coord="y", iteration=iteration, plot=False)
    Bz, Bz_info = ts.get_field(field="B", coord="z", iteration=iteration, plot=False)
    rho_1, _ = ts.get_field(field="rho_electrons_1", iteration=iteration, plot=False)

    if grid_shape is None:
        grid_shape = Bz.shape
        Bz_mode_power = np.empty((t.size,) + grid_shape)
    assert Bz.shape == grid_shape, (
        f"Snapshot at iteration {iteration} has grid shape {Bz.shape}, expected {grid_shape}: "
        "diags/diag1 probably contains output from an earlier run on a different grid"
    )

    Bx_2[i] = np.mean(np.square(Bx))
    By_2[i] = np.mean(np.square(By))
    Bz_2[i] = np.mean(np.square(Bz))
    Bz_mode_power[i] = np.abs(np.fft.fft2(Bz)) ** 2 / Bz.size**2
    Bz_abs_max = max(Bz_abs_max, float(np.max(np.abs(Bz))))
    if rho_spatial_mean is None:
        rho_spatial_mean = float(np.mean(rho_1))
    rho_dev_max = max(rho_dev_max, float(np.max(np.abs(rho_1 - rho_spatial_mean))))

assert np.all(np.isfinite(Bx_2))
assert np.all(np.isfinite(By_2))
assert np.all(np.isfinite(Bz_2))
assert Bz_abs_max > 0.0, "B_z is identically zero across all field diagnostics"

# FieldEnergy reduced diagnostic. Columns: step, time, total, electric, magnetic energy.
field_energy = np.atleast_2d(np.loadtxt("diags/reducedfiles/EF.txt", skiprows=1))
assert field_energy.shape[0] >= 2
assert field_energy.shape[1] == 5
assert np.all(np.isfinite(field_energy))
assert np.all(field_energy[:, 2:] >= 0.0)

step = field_energy[:, 0]
field_time = field_energy[:, 1]
total_energy = field_energy[:, 2]
electric_energy = field_energy[:, 3]
magnetic_energy = field_energy[:, 4]

assert np.all(np.diff(step) > 0)
assert np.all(np.diff(field_time) >= 0.0)
assert np.all(np.isfinite(total_energy))
assert np.all(np.isfinite(electric_energy))
assert np.all(np.isfinite(magnetic_energy))

# Rough check: the magnetic energy is higher, on average, in the second half of the run.
early_magnetic_energy = np.mean(magnetic_energy[: field_energy.shape[0] // 2])
late_magnetic_energy = np.mean(magnetic_energy[field_energy.shape[0] // 2 :])
assert late_magnetic_energy > early_magnetic_energy

# The peak magnetic energy must clearly exceed its initial level (seed plus noise), averaged over
# the start of the run.
n_early = min(len(magnetic_energy), max(2, len(magnetic_energy) // 20))
early_reference = np.mean(magnetic_energy[:n_early])
assert magnetic_energy.max() > 2.0 * early_reference, (
    "Magnetic energy did not grow appreciably over the run: "
    f"max/early = {magnetic_energy.max() / max(early_reference, np.finfo(float).tiny):.3g}"
)

# At the last snapshot, the total rho must equal the sum of the species' charge densities.
rho_1, _ = ts.get_field(field="rho_electrons_1", iteration=iterations[-1], plot=False)
rho_2, _ = ts.get_field(field="rho_electrons_2", iteration=iterations[-1], plot=False)
rho_i, _ = ts.get_field(field="rho_ions", iteration=iterations[-1], plot=False)
rho, _ = ts.get_field(field="rho", iteration=iterations[-1], plot=False)

assert np.all(np.isfinite(rho_1))
assert np.all(np.isfinite(rho_2))
assert np.all(np.isfinite(rho_i))
assert np.all(np.isfinite(rho))
# Typical charge density, for the absolute tolerance.
charge_scale = abs(q_e) * n_0
np.testing.assert_allclose(
    rho, rho_1 + rho_2 + rho_i, rtol=1.0e-7, atol=1.0e-10 * charge_scale
)

# At late times, B_z must be the largest component (averaged over the last few snapshots).
# B_z > B_y follows from the physics; B_z > B_x from the setup (dx < dz, seed mode along x).
late = slice(max(1, len(t) - 5), len(t))
Bx_late = np.mean(Bx_2[late])
By_late = np.mean(By_2[late])
Bz_late = np.mean(Bz_2[late])
dominance_factor = 1.0
assert Bz_late > dominance_factor * max(Bx_late, By_late), (
    "Expected late-time B_z^2 to be the leading magnetic component: "
    f"B_z^2={Bz_late:.6e}, B_x^2={Bx_late:.6e}, B_y^2={By_late:.6e}"
)


def cold_growth_rate(k_d):
    """Cold-beam Weibel growth rate at wavenumber k = k_d / d, with d = c / omega_p.

    For k perpendicular to the drift, gamma(k) = beta * omega_p * k d / sqrt(1 + (k d)^2), which
    tends to beta * omega_p for k d >> 1. Reference: E. S. Weibel, Phys. Rev. Lett. 2, 83 (1959),
    Eq. (9) and its approximate root omega_4 (there with c = 1 and u_0^2 = v_drift^2 + v_thermal^2
    ~ (beta c)^2). It holds for any velocity distribution with a small thermal spread along k
    (k v_thermal << gamma).
    """
    return gamma_theory * k_d / np.sqrt(1.0 + k_d**2)


# Fourier index of the seed mode: seed_mode along x, 0 along z.
axis_names = [Bz_info.axes[axis] for axis in range(2)]
assert sorted(axis_names) == ["x", "z"], f"Unexpected grid axes {axis_names}"
assert grid_shape[axis_names.index("x")] == n_cell_x
seed_index = tuple(seed_mode if name == "x" else 0 for name in axis_names)
# B_z is real, so modes k and -k together form sin(k x): their summed power is the seed-mode
# share of <B_z^2>.
partner_index = tuple((-i) % n for i, n in zip(seed_index, grid_shape))
mode_power = (
    Bz_mode_power[(slice(None),) + seed_index]
    + Bz_mode_power[(slice(None),) + partner_index]
)
# Zero power gives -inf; such snapshots are left out of the fit.
with np.errstate(divide="ignore"):
    ln_mode_power = np.log(mode_power)
gamma_theory_mode = cold_growth_rate(seed_k_d)
print(f"Cold gamma(k) of seed mode: {gamma_theory_mode:.6e} 1/s")

# Fit window: from omega_p * t = FIT_START_OMEGA_P_T (after the start-up phase) to the last
# snapshot. The run stops before saturation, so the mode is still growing exponentially there.
# A fixed start time keeps the window independent of the output interval.
candidate = np.flatnonzero(
    np.isfinite(ln_mode_power) & (omega_p * t >= FIT_START_OMEGA_P_T)
)
assert candidate.size >= MIN_FIT_POINTS, (
    "The seed mode did not grow past the initial transient: "
    f"{candidate.size} usable diagnostics after omega_p * t = {FIT_START_OMEGA_P_T}, "
    f"t_final={t[-1]:.6e} s"
)

# Use one uninterrupted window, with enough snapshots and a clear rise of the mode power.
fit_indices = largest_contiguous_block(candidate)
fit_ln_rise = ln_mode_power[fit_indices[-1]] - ln_mode_power[fit_indices[0]]
assert fit_indices.size >= MIN_FIT_POINTS and fit_ln_rise >= MIN_FIT_LN_BZ_2_RISE, (
    "The growth window of the seed mode after the initial transient is too short: "
    f"{fit_indices.size} diagnostics and {fit_ln_rise:.3g} rise in ln(power) "
    f"between t={t[fit_indices[0]]:.6e} s and t={t[fit_indices[-1]]:.6e} s."
)

fit_t = t[fit_indices]
fit_ln_mode_power = ln_mode_power[fit_indices]
slope, intercept, r_sq = linear_fit(fit_t, fit_ln_mode_power)
# The power grows as exp(2 * gamma * t), so gamma is half the slope.
gamma_fit = 0.5 * slope

# For comparison only (not checked): growth rate of <B_z^2> over the same window, which includes
# the noise modes.
with np.errstate(divide="ignore"):
    ln_Bz_2 = np.log(Bz_2)
gamma_grid_average = 0.5 * linear_fit(fit_t, ln_Bz_2[fit_indices])[0]

print(f"Fit window: {fit_t[0]:.6e} s to {fit_t[-1]:.6e} s")
print(f"Fit points: {fit_indices.size}")
print(f"ln(mode power) rise over fit window: {fit_ln_rise:.6g}")
print(
    "Seed-mode share of <B_z^2> at start and end of fit window: "
    f"{mode_power[fit_indices[0]] / Bz_2[fit_indices[0]]:.3g}, "
    f"{mode_power[fit_indices[-1]] / Bz_2[fit_indices[-1]]:.3g}"
)
print(f"Fit R^2: {r_sq:.6f}")
print(f"Fitted gamma of seed mode: {gamma_fit:.6e} 1/s")
print(f"gamma_fit / gamma(k): {gamma_fit / gamma_theory_mode:.6f}")
print(f"gamma_fit / (beta * omega_p): {gamma_fit / gamma_theory:.6f}")
print(
    "Growth rate of grid-averaged B_z^2 / gamma(k) (not checked): "
    f"{gamma_grid_average / gamma_theory_mode:.6f}"
)

# Theory line with its intercept matched to the data in the fit window, to compare slopes by eye.
theory_intercept = np.mean(fit_ln_mode_power - 2.0 * gamma_theory_mode * fit_t)
theory_ln = theory_intercept + 2.0 * gamma_theory_mode * t
fit_ln = intercept + slope * t

# Plot <B_i^2>, the seed-mode power, and its theoretical and fitted growth.
fig, ax = plt.subplots()
ax.semilogy(t, Bx_2, label=r"$\langle B_x^2 \rangle$")
ax.semilogy(t, By_2, label=r"$\langle B_y^2 \rangle$")
ax.semilogy(t, Bz_2, label=r"$\langle B_z^2 \rangle$")
ax.semilogy(
    t,
    mode_power,
    label=rf"$B_z$ seed mode, $k d={seed_k_d:.2f}$",
)
ax.semilogy(
    t, np.exp(theory_ln), "--", label=r"$\propto e^{2 \gamma(k) t}$ (cold theory)"
)
ax.semilogy(t, np.exp(fit_ln), ":", label="fit to mode")
ax.axvspan(fit_t[0], fit_t[-1], color="0.85", label="fit window")

# Vertical range from the positive plotted values (zeros cannot be shown on a log scale).
positive_plot_values = np.concatenate(
    (
        Bx_2[Bx_2 > 0.0],
        By_2[By_2 > 0.0],
        Bz_2[Bz_2 > 0.0],
        mode_power[mode_power > 0.0],
        np.exp(theory_ln),
        np.exp(fit_ln),
    )
)
positive_plot_values = positive_plot_values[np.isfinite(positive_plot_values)]
assert positive_plot_values.size > 0

ax.set_title(f"Weibel magnetic-field growth ({solver_name})")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\langle B_i^2 \rangle$ (T$^2$)")
ax.set_xlim(t[0], t[-1])
ax.set_ylim(0.5 * positive_plot_values.min(), 2.0 * positive_plot_values.max())
ax.legend(loc="best", fontsize="small")
fig.tight_layout()
fig.savefig("weibel_magnetic_energy_growth.png", dpi=150)
plt.close(fig)

# Animations of B_z and of the electrons_1 charge density. For rho, white is the initial mean,
# so filaments show up as red and blue regions.
save_field_animation(
    iterations,
    t,
    get_field_frame=lambda it: ts.get_field(
        field="B", coord="z", iteration=it, plot=False
    ),
    vmin=-Bz_abs_max,
    vmax=Bz_abs_max,
    colorbar_label=r"$B_z$ (T)",
    title_prefix="B_z",
    filename="weibel_Bz.gif",
)
save_field_animation(
    iterations,
    t,
    get_field_frame=lambda it: ts.get_field(
        field="rho_electrons_1", iteration=it, plot=False
    ),
    vmin=rho_spatial_mean - rho_dev_max,
    vmax=rho_spatial_mean + rho_dev_max,
    colorbar_label=r"$\rho_{\mathrm{e1}}$ (C/m$^3$)",
    title_prefix="rho_electrons_1",
    filename="weibel_rho_electrons_1.gif",
)

# Check the fit after plotting, so that the plots exist even if a check fails: the growth must
# be exponential (good linear fit) at a rate close to the cold-beam gamma(k).
assert np.isfinite(gamma_fit)
assert gamma_fit > 0.0
assert r_sq > 0.80
assert gamma_fit <= gamma_theory_mode * (1.0 + GAMMA_RTOL_UPPER), (
    "Fitted growth rate of the seed mode exceeds the cold-beam upper bound gamma(k): "
    f"gamma_fit={gamma_fit:.6e}, gamma(k)={gamma_theory_mode:.6e}"
)
assert gamma_fit >= gamma_theory_mode * (1.0 - GAMMA_RTOL_LOWER), (
    "Fitted growth rate of the seed mode is lower than expected for the resolved linear "
    f"phase: gamma_fit={gamma_fit:.6e}, gamma(k)={gamma_theory_mode:.6e}"
)
