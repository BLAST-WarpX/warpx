#!/usr/bin/env python3

# Copyright 2023-2026 Juliette Pech, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Analyse the linear phase of the 2D Weibel instability test.

The two electron populations counter-stream along y. In this geometry the unstable magnetic
field component is B_z, so the growth-rate fit is performed on ln(sum(B_z^2)). The fitted
growth rate is compared to the cold k -> infinity asymptote gamma = beta * omega_p, used here
as an upper-bound estimate for the warm, finite-box simulation.
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
from scipy.constants import e as q_e
from scipy.constants import epsilon_0, m_e

MIN_FIT_POINTS = 6  # minimum number of diagnostics needed to fit a growth rate
GAMMA_RTOL_UPPER = 0.10  # fitted gamma must not exceed gamma_theory by more than 10%
GAMMA_RTOL_LOWER = (
    0.30  # fitted gamma must not fall below gamma_theory by more than 30%
)


def largest_contiguous_block(indices):
    """Return the longest run of consecutive integers from *indices*."""
    assert indices.size > 0
    # Find positions where the gap between successive indices is > 1, then split there.
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    blocks = np.split(indices, split_points)
    return max(blocks, key=len)


def linear_fit(x, y):
    """Fit y = slope*x + intercept and return (slope, intercept, R^2)."""
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)  # residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    r_sq = (
        1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    )  # coefficient of determination
    return slope, intercept, r_sq


def save_field_animation(
    iterations, t, get_field_frame, abs_max, colorbar_label, title_prefix, filename
):
    """Create and save an animated GIF for a 2-D field quantity.

    Parameters
    ----------
    iterations : array of int
    t : array of float
    get_field_frame : callable(iteration) -> (data, info)
    abs_max : float — symmetric color-scale limit; falls back to 1.0 when zero
    colorbar_label : str
    title_prefix : str — prepended to "iteration N - time T s"
    filename : str — output GIF path
    """
    fig, ax = plt.subplots()
    data, info = get_field_frame(iterations[0])
    clim = abs_max if abs_max > 0.0 else 1.0
    im = ax.imshow(
        data,
        origin="lower",
        extent=info.imshow_extent,
        aspect="auto",
        interpolation="nearest",
        vmin=-clim,
        vmax=clim,
        cmap="RdBu_r",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    title = ax.set_title(
        f"{title_prefix}, iteration {iterations[0]} - time {t[0]:.2e} s"
    )
    ax.set_xlabel(f"{info.axes[1]} (m)")
    ax.set_ylabel(f"{info.axes[0]} (m)")

    def update(i):
        iteration = iterations[i]
        data, info = get_field_frame(iteration)
        im.set_data(data)
        im.set_extent(info.imshow_extent)
        title.set_text(f"{title_prefix}, iteration {iteration} - time {t[i]:.2e} s")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(iterations), interval=100)
    ani.save(filename, writer="pillow")
    plt.close(fig)


def load_input_deck():
    input_file = Path("warpx_used_inputs")
    if not input_file.exists():
        input_file = Path("inputs_test_2d_weibel_instability")
    assert input_file.exists(), (
        "Could not find warpx_used_inputs or the local input deck"
    )
    return parse_input_file(input_file)


# Parse simulation parameters at module level so they are available to all subsequent code.
input_dict = load_input_deck()

species_names = input_dict["particles.species_names"]
assert len(species_names) == 2, f"Expected two electron species, got {species_names}"
assert species_names == ["electrons_1", "electrons_2"]

if "geometry.dims" in input_dict:
    assert input_dict["geometry.dims"][0] == "2"

# Read species parameters: beta = v_drift/c, theta = kT/(m_e c^2) (thermal spread).
betas = np.array([float(input_dict[f"{species}.beta"][0]) for species in species_names])
densities = np.array(
    [float(input_dict[f"{species}.density"][0]) for species in species_names]
)
thetas = np.array(
    [float(input_dict[f"{species}.theta"][0]) for species in species_names]
)
drift_dirs = [input_dict[f"{species}.bulk_vel_dir"][0] for species in species_names]
charges = [input_dict[f"{species}.charge"][0] for species in species_names]
masses = [input_dict[f"{species}.mass"][0] for species in species_names]

# Enforce the symmetric counter-streaming electron configuration required by this test.
assert set(drift_dirs) == {"y", "-y"}, (
    f"Expected counter-streaming y beams, got {drift_dirs}"
)
assert charges == ["-q_e", "-q_e"]
assert masses == ["m_e", "m_e"]
assert np.all(betas > 0.0)
assert np.all(densities > 0.0)
assert np.all(thetas > 0.0)
# Both populations must be identical so the combined plasma is symmetric.
np.testing.assert_allclose(betas, betas[0])
np.testing.assert_allclose(densities, densities[0])
np.testing.assert_allclose(thetas, thetas[0])

beta = float(np.mean(betas))  # common drift speed v_d/c
n_0 = float(np.sum(densities))  # total electron number density
omega_p = math.sqrt(
    n_0 * q_e**2 / (m_e * epsilon_0)
)  # non-relativistic electron plasma frequency
# Cold-beam k -> infinity asymptote: gamma = beta * omega_p (used as an upper bound).
gamma_theory = beta * omega_p

max_step = int(input_dict["max_step"][0]) if "max_step" in input_dict else None

print(f"Plasma frequency omega_p: {omega_p:.6e} 1/s")
print(f"Cold asymptotic gamma upper bound: {gamma_theory:.6e} 1/s")
print(f"v_drift / v_thermal: {beta / math.sqrt(float(np.mean(thetas))):.6g}")

# Load the openPMD time series written by the WarpX field diagnostics.
ts = OpenPMDTimeSeries("diags/diag1")
iterations = np.asarray(ts.iterations)  # integer step numbers for each snapshot
t = np.asarray(ts.t)  # physical times in seconds
assert iterations.size == t.size
assert t.size >= MIN_FIT_POINTS, "Need at least 6 field diagnostics to fit growth"
assert np.all(np.diff(t) > 0.0)  # times must be strictly increasing

print(f"Final diagnostic time: {t[-1]:.6e} s")
if max_step is not None:
    assert iterations[-1] <= max_step, (
        "The diagnostics do not appear to come from the parsed input file: "
        f"final iteration {iterations[-1]} exceeds max_step {max_step}"
    )

# Accumulate sum(B_i^2) over all cells at every diagnostic iteration.
B_x_2 = np.empty_like(t)
B_y_2 = np.empty_like(t)
B_z_2 = np.empty_like(t)
B_z_abs_max = 0.0  # track the peak |B_z| for the animation color scale
rho_abs_max = 0.0  # track the peak |rho_electrons_1| for the animation color scale

for i, iteration in enumerate(iterations):
    B_x, _ = ts.get_field(field="B", coord="x", iteration=iteration, plot=False)
    B_y, _ = ts.get_field(field="B", coord="y", iteration=iteration, plot=False)
    B_z, _ = ts.get_field(field="B", coord="z", iteration=iteration, plot=False)
    rho_frame, _ = ts.get_field(
        field="rho_electrons_1", iteration=iteration, plot=False
    )

    B_x_2[i] = np.sum(np.square(B_x))
    B_y_2[i] = np.sum(np.square(B_y))
    B_z_2[i] = np.sum(np.square(B_z))
    B_z_abs_max = max(B_z_abs_max, float(np.max(np.abs(B_z))))
    rho_abs_max = max(rho_abs_max, float(np.max(np.abs(rho_frame))))

assert np.all(np.isfinite(B_x_2))
assert np.all(np.isfinite(B_y_2))
assert np.all(np.isfinite(B_z_2))
assert B_z_abs_max > 0.0, "B_z is identically zero across all field diagnostics"

# Load the FieldEnergy reduced diagnostic. Columns: 0=step, 1=time, 2=total, 3=electric, 4=magnetic energy.
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

# Coarse check: magnetic energy must be higher in the second half of the run than the first.
early_magnetic_energy = np.mean(magnetic_energy[: field_energy.shape[0] // 2])
late_magnetic_energy = np.mean(magnetic_energy[field_energy.shape[0] // 2 :])
assert late_magnetic_energy > early_magnetic_energy

# Use the first ~5% of steps as the noise reference to confirm growth by at least 2x.
n_early = min(len(magnetic_energy), max(2, len(magnetic_energy) // 20))
early_reference = np.mean(magnetic_energy[:n_early])
assert magnetic_energy.max() > 2.0 * early_reference, (
    "Magnetic energy did not grow appreciably over the run: "
    f"max/early = {magnetic_energy.max() / max(early_reference, np.finfo(float).tiny):.3g}"
)

# Verify charge-density consistency: total rho is the sum of the two electron depositions.
rho_1, _ = ts.get_field(field="rho_electrons_1", iteration=iterations[-1], plot=False)
rho_2, _ = ts.get_field(field="rho_electrons_2", iteration=iterations[-1], plot=False)
rho, _ = ts.get_field(field="rho", iteration=iterations[-1], plot=False)

assert np.all(np.isfinite(rho_1))
assert np.all(np.isfinite(rho_2))
assert np.all(np.isfinite(rho))
charge_scale = (
    abs(q_e) * n_0
)  # characteristic charge density used for absolute tolerance
np.testing.assert_allclose(rho, rho_1 + rho_2, rtol=1.0e-7, atol=1.0e-10 * charge_scale)

# The out-of-plane component B_z is the Weibel-unstable mode for k || x and v_d || y.
# Confirm that it dominates the other components at late time.
late = slice(max(1, len(t) - 5), len(t))  # average over the last 5 snapshots
B_x_late = np.mean(B_x_2[late])
B_y_late = np.mean(B_y_2[late])
B_z_late = np.mean(B_z_2[late])
dominance_factor = 1.0
assert B_z_late > dominance_factor * max(B_x_late, B_y_late), (
    "Expected late-time B_z^2 to be the leading magnetic component: "
    f"B_z^2={B_z_late:.6e}, B_x^2={B_x_late:.6e}, B_y^2={B_y_late:.6e}"
)

# Select the linear growth phase by amplitude, excluding early startup noise and late saturation.
with np.errstate(divide="ignore", invalid="ignore"):
    ln_B_z_2 = np.log(
        B_z_2
    )  # work in log-space; B_z_2 == 0 gives -inf (silently ignored)

finite_positive = np.flatnonzero(np.isfinite(ln_B_z_2))
assert finite_positive.size >= MIN_FIT_POINTS, (
    f"Need at least {MIN_FIT_POINTS} positive B_z^2 diagnostics to fit growth"
)

# Estimate the noise floor from the first few finite snapshots.
noise_sample_count = min(5, max(2, finite_positive.size // 10))
noise_indices = finite_positive[:noise_sample_count]
noise_floor = np.mean(ln_B_z_2[noise_indices])
saturation_level = np.max(ln_B_z_2[finite_positive])  # log-amplitude at saturation
dynamic_range = saturation_level - noise_floor  # total e-folds of growth

assert dynamic_range > 3.0, (
    f"Dynamic range of B_z^2 is only {dynamic_range:.3g} e-folds; "
    "the linear Weibel phase did not develop enough to fit a growth rate."
)

# Keep only the window that is at least 2 e-folds above noise and 1 e-fold below saturation.
candidate = np.flatnonzero(
    np.isfinite(ln_B_z_2)
    & (ln_B_z_2 >= noise_floor + 2.0)
    & (ln_B_z_2 <= saturation_level - 1.0)
)
assert candidate.size >= MIN_FIT_POINTS, (
    "Could not identify a resolved linear B_z^2 growth window. "
    f"t_final={t[-1]:.6e} s, ln-noise floor={noise_floor:.6g}, "
    f"ln-saturation={saturation_level:.6g}"
)

fit_indices = largest_contiguous_block(candidate)
assert fit_indices.size >= MIN_FIT_POINTS, (
    "The amplitude-selected B_z^2 growth window is too short. "
    f"Selected {fit_indices.size} diagnostics between t={t[fit_indices[0]]:.6e} s "
    f"and t={t[fit_indices[-1]]:.6e} s."
)

fit_t = t[fit_indices]
fit_ln_B_z_2 = ln_B_z_2[fit_indices]
slope, intercept, r_sq = linear_fit(fit_t, fit_ln_B_z_2)
# ln(B_z^2) grows as 2*gamma*t, so the growth rate is half the log-slope.
gamma_fit = 0.5 * slope

# Amplitude-match the theoretical slope to the fit window for plotting.
theory_intercept = np.mean(fit_ln_B_z_2 - 2.0 * gamma_theory * fit_t)
theory_ln = (
    theory_intercept + 2.0 * gamma_theory * t
)  # extended over all t for the plot
fit_ln = intercept + slope * t

print(f"Fit window: {fit_t[0]:.6e} s to {fit_t[-1]:.6e} s")
print(f"Fit points: {fit_indices.size}")
print(f"B_z^2 dynamic range: {dynamic_range:.6g} e-folds")
print(f"Fit R^2: {r_sq:.6f}")
print(f"Fitted gamma: {gamma_fit:.6e} 1/s")
print(f"gamma_fit / gamma_theory: {gamma_fit / gamma_theory:.6f}")

assert np.isfinite(gamma_fit)
assert gamma_fit > 0.0
assert r_sq > 0.80
assert gamma_fit <= gamma_theory * (1.0 + GAMMA_RTOL_UPPER), (
    "Fitted growth rate exceeds the cold asymptotic upper-bound estimate: "
    f"gamma_fit={gamma_fit:.6e}, gamma_theory={gamma_theory:.6e}"
)
assert gamma_fit >= gamma_theory * (1.0 - GAMMA_RTOL_LOWER), (
    "Fitted growth rate is lower than expected for the resolved linear phase: "
    f"gamma_fit={gamma_fit:.6e}, gamma_theory={gamma_theory:.6e}"
)

# Semilog plot of all three magnetic-field components together with the theory and fit slopes.
fig, ax = plt.subplots()
ax.semilogy(t, B_x_2, label=r"$\sum B_x^2$")
ax.semilogy(t, B_y_2, label=r"$\sum B_y^2$")
ax.semilogy(t, B_z_2, label=r"$\sum B_z^2$ (unstable)")
ax.semilogy(
    t,
    np.exp(theory_ln),
    "--",
    label=r"cold asymptotic slope, amplitude matched",
)
ax.semilogy(t, np.exp(fit_ln), ":", label=r"fit to $\sum B_z^2$")
ax.axvspan(
    fit_t[0], fit_t[-1], color="0.85", label="fit window"
)  # shade the fitted interval

positive_plot_values = np.concatenate(
    (
        B_x_2[B_x_2 > 0.0],
        B_y_2[B_y_2 > 0.0],
        B_z_2[B_z_2 > 0.0],
        np.exp(theory_ln),
        np.exp(fit_ln),
    )
)
positive_plot_values = positive_plot_values[np.isfinite(positive_plot_values)]
assert positive_plot_values.size > 0

ax.set_title("Weibel magnetic-field growth")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\sum B_i^2$")
ax.set_xlim(t[0], t[-1])
ax.set_ylim(0.5 * positive_plot_values.min(), 2.0 * positive_plot_values.max())
ax.legend(loc="best", fontsize="small")
fig.tight_layout()
fig.savefig("weibel_magnetic_energy_growth.png", dpi=150)
plt.close(fig)

# Animate the unstable B_z component and the electrons_1 charge density.
save_field_animation(
    iterations,
    t,
    get_field_frame=lambda it: ts.get_field(
        field="B", coord="z", iteration=it, plot=False
    ),
    abs_max=B_z_abs_max,
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
    abs_max=rho_abs_max,
    colorbar_label=r"$\rho_{\mathrm{e1}}$ (C/m$^3$)",
    title_prefix="rho_electrons_1",
    filename="weibel_rho_electrons_1.gif",
)
