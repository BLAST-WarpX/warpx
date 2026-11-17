#!/usr/bin/env python3

# Copyright 2023-2026 Juliette Pech, Edoardo Zoni
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
Check the early, exponential-growth phase of the 2D Weibel instability test.

Physical setup: two electron beams move in opposite directions along y, which is perpendicular
to the x-z simulation plane. The Weibel instability then produces magnetic-field structures in
the x-z plane:
- structures that vary along x produce B_z,
- structures that vary along z produce B_x,
- B_y is not produced and stays at the level of the particle noise.
In the cold-beam limit, B_x and B_z grow equally fast. However, the grid spacing is finer in x
than in z (dx < dz), so the grid resolves more structures that vary along x. As a result, B_z
holds most of the magnetic energy, and we use B_z to measure the growth rate.

How the growth rate is measured: while the instability grows exponentially, B_z^2 summed over
the grid is proportional to exp(2 * gamma * t). Its logarithm is therefore a straight line in
time with slope 2 * gamma. We fit that line and compare gamma to beta * omega_p, which is the
growth rate predicted for cold beams at very short wavelengths. Since the simulated beams are
warm and the box is finite, the simulation should grow somewhat slower than this, so beta *
omega_p serves as an upper bound.
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

# Need at least this many field snapshots to fit a growth rate.
MIN_FIT_POINTS = 6
# The fit starts once omega_p * t reaches this value. Earlier times are the start-up phase, when
# the growing structures are still emerging from the particle noise.
FIT_START_OMEGA_P_T = 75.0
# Over the fit window, ln(B_z^2) must increase by at least this much (i.e., B_z^2 must grow by at
# least a factor e), so that the fit measures actual growth and not noise.
MIN_FIT_LN_BZ_2_RISE = 1.0
# The fitted growth rate may be at most 10% above the theoretical upper bound and at most 30% below it.
GAMMA_RTOL_UPPER = 0.10
GAMMA_RTOL_LOWER = 0.30


def largest_contiguous_block(indices):
    """Keep only the longest stretch of snapshots that follow each other without gaps.

    *indices* is a sorted array of snapshot numbers. For example, [2, 3, 7, 8, 9] has two
    stretches, [2, 3] and [7, 8, 9], and this function returns [7, 8, 9]. This makes sure the
    growth-rate fit uses one uninterrupted time window.
    """
    assert indices.size > 0
    # A gap is where two neighbouring numbers differ by more than 1: cut the array there.
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    blocks = np.split(indices, split_points)
    return max(blocks, key=len)


def linear_fit(x, y):
    """Fit a straight line y = slope * x + intercept through the data points.

    Returns slope, intercept, and R^2. R^2 measures how well the line matches the data:
    1 means the points lie exactly on the line, values near 0 mean the line explains almost
    nothing.
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    # Squared distance of the points from the fitted line...
    ss_res = np.sum((y - y_fit) ** 2)
    # ...compared to their squared distance from their own average.
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r_sq


def save_field_animation(
    iterations, t, get_field_frame, vmin, vmax, colorbar_label, title_prefix, filename
):
    """Save a GIF that shows how a 2D field evolves, one frame per snapshot.

    Parameters
    ----------
    iterations : array of int
        Simulation step number of each snapshot.
    t : array of float
        Physical time of each snapshot, in seconds.
    get_field_frame : function
        Takes a step number and returns (data, info) for that snapshot, as returned by
        openPMD-viewer's ``get_field``.
    vmin, vmax : float
        Values mapped to the two ends of the color scale. They are the same for all frames, so
        that colors can be compared between frames.
    colorbar_label : str
        Label of the color bar.
    title_prefix : str
        Start of each frame's title; the step number and time are appended to it.
    filename : str
        Name of the GIF file to write.
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
    """Read the simulation input parameters into a dictionary.

    Prefer ``warpx_used_inputs``, which WarpX writes next to the output and which contains the
    parameters that were actually used. If it is missing, fall back to the original input file.
    """
    input_file = Path("warpx_used_inputs")
    if not input_file.exists():
        input_file = Path("inputs_test_2d_weibel_instability")
    assert input_file.exists(), (
        "Could not find warpx_used_inputs or the local input deck"
    )
    return parse_input_file(input_file)


# Read the simulation parameters once; the rest of the script uses them.
input_dict = load_input_deck()

species_names = input_dict["particles.species_names"]
electron_species = ["electrons_1", "electrons_2"]
ion_species = "ions"
assert species_names == electron_species + [ion_species], (
    f"Expected two electron species and a neutralizing ion background, got {species_names}"
)

if "geometry.dims" in input_dict:
    assert input_dict["geometry.dims"][0] == "2"

# Read the electron beam parameters. The input file gives momenta as u = gamma * beta (momentum
# divided by m_e * c), with gamma = 1 / sqrt(1 - beta**2) the Lorentz factor. The mean of u is
# the beam drift; its spread (standard deviation) is the thermal spread, equal to sqrt(theta),
# where theta = k_B T / (m_e c^2) is the temperature in units of the electron rest energy.
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
# Convert u = gamma * beta back to the velocity beta = v / c.
beta_means = u_means / np.sqrt(1.0 + u_means**2)
# Drift speed of each beam along y.
betas = np.abs(beta_means[:, 1])
densities = np.array(
    [float(input_dict[f"{species}.density"][0]) for species in electron_species]
)
u_stds = np.array(
    [
        [
            float(input_dict[f"{species}.{component}_std"][0])
            for component in momentum_components
        ]
        for species in electron_species
    ]
)
# The thermal spread must be the same in x, y, and z (same temperature in all directions).
np.testing.assert_allclose(u_stds[:, 1], u_stds[:, 0])
np.testing.assert_allclose(u_stds[:, 2], u_stds[:, 0])
thetas = u_stds[:, 0] ** 2
drift_dirs = ["y" if beta_mean[1] > 0.0 else "-y" for beta_mean in beta_means]
charges = [input_dict[f"{species}.charge"][0] for species in electron_species]
masses = [input_dict[f"{species}.mass"][0] for species in electron_species]

# This test expects two electron beams moving in opposite directions along y.
assert set(drift_dirs) == {"y", "-y"}, (
    f"Expected counter-streaming y beams, got {drift_dirs}"
)
assert charges == ["-q_e", "-q_e"]
assert masses == ["m_e", "m_e"]
assert np.all(betas > 0.0)
assert np.all(densities > 0.0)
assert np.all(thetas > 0.0)
# The two beams must have the same speed, density, and temperature, so that they mirror each
# other and the plasma as a whole carries no net current.
np.testing.assert_allclose(betas, betas[0])
np.testing.assert_allclose(densities, densities[0])
np.testing.assert_allclose(thetas, thetas[0])

# Drift speed of the beams, as a fraction of the speed of light.
beta = float(np.mean(betas))
# Total electron density of both beams together.
n_0 = float(np.sum(densities))

# The Darwin field solver computes the electric potential on a periodic domain, which only works
# if the total charge is zero. The ions therefore form a uniform background that does not move
# and exactly cancels the electron charge.
assert input_dict[f"{ion_species}.charge"][0] == "q_e"
assert input_dict[f"{ion_species}.do_not_push"][0] == "1"
assert input_dict[f"{ion_species}.momentum_distribution_type"][0] == "at_rest"
np.testing.assert_allclose(float(input_dict[f"{ion_species}.density"][0]), n_0)
# Electron plasma frequency (without relativistic corrections).
omega_p = math.sqrt(n_0 * q_e**2 / (m_e * epsilon_0))
# Theoretical growth rate for cold beams at very short wavelengths. Warm beams in a finite box
# grow somewhat slower, so this is used as an upper bound.
gamma_theory = beta * omega_p

max_step = int(input_dict["max_step"][0]) if "max_step" in input_dict else None

print(f"Plasma frequency omega_p: {omega_p:.6e} 1/s")
print(f"Cold asymptotic gamma upper bound: {gamma_theory:.6e} 1/s")
print(f"v_drift / v_thermal: {beta / math.sqrt(float(np.mean(thetas))):.6g}")

# Open the field snapshots written by the simulation.
ts = OpenPMDTimeSeries("diags/diag1")
# Simulation step number of each snapshot.
iterations = np.asarray(ts.iterations)
# Physical time of each snapshot, in seconds.
t = np.asarray(ts.t)
assert iterations.size == t.size
assert t.size >= MIN_FIT_POINTS, "Need at least 6 field diagnostics to fit growth"
# Each snapshot must come later in time than the previous one.
assert np.all(np.diff(t) > 0.0)

print(f"Final diagnostic time: {t[-1]:.6e} s")
# Guard against analysing output left over from a different (longer) run.
if max_step is not None:
    assert iterations[-1] <= max_step, (
        "The diagnostics do not appear to come from the parsed input file: "
        f"final iteration {iterations[-1]} exceeds max_step {max_step}"
    )

# For each snapshot, add up B_x^2, B_y^2, and B_z^2 over all grid cells. These sums are
# proportional to the magnetic energy in each component.
Bx_2 = np.empty_like(t)
By_2 = np.empty_like(t)
Bz_2 = np.empty_like(t)
# Largest |B_z| over all snapshots; sets the color scale of the B_z animation.
Bz_abs_max = 0.0
# Average charge density of the electrons_1 beam in the first snapshot, i.e., before any
# filaments have formed. The rho animation is colored relative to this value.
rho_spatial_mean = None
# Largest local difference from that average, over all snapshots; sets the color scale of the
# rho animation.
rho_dev_max = 0.0

for i, iteration in enumerate(iterations):
    Bx, _ = ts.get_field(field="B", coord="x", iteration=iteration, plot=False)
    By, _ = ts.get_field(field="B", coord="y", iteration=iteration, plot=False)
    Bz, _ = ts.get_field(field="B", coord="z", iteration=iteration, plot=False)
    rho_1, _ = ts.get_field(field="rho_electrons_1", iteration=iteration, plot=False)

    Bx_2[i] = np.sum(np.square(Bx))
    By_2[i] = np.sum(np.square(By))
    Bz_2[i] = np.sum(np.square(Bz))
    Bz_abs_max = max(Bz_abs_max, float(np.max(np.abs(Bz))))
    if rho_spatial_mean is None:
        rho_spatial_mean = float(np.mean(rho_1))
    rho_dev_max = max(rho_dev_max, float(np.max(np.abs(rho_1 - rho_spatial_mean))))

assert np.all(np.isfinite(Bx_2))
assert np.all(np.isfinite(By_2))
assert np.all(np.isfinite(Bz_2))
assert Bz_abs_max > 0.0, "B_z is identically zero across all field diagnostics"

# Load the field energy over time, written by the FieldEnergy reduced diagnostic. Its columns
# are: step, time, total field energy, electric energy, magnetic energy.
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

# Rough check: on average, the magnetic energy must be higher in the second half of the run
# than in the first half.
early_magnetic_energy = np.mean(magnetic_energy[: field_energy.shape[0] // 2])
late_magnetic_energy = np.mean(magnetic_energy[field_energy.shape[0] // 2 :])
assert late_magnetic_energy > early_magnetic_energy

# The average over the first ~5% of the run gives the initial (noise) level of the magnetic
# energy. Its peak value must be at least twice that level.
n_early = min(len(magnetic_energy), max(2, len(magnetic_energy) // 20))
early_reference = np.mean(magnetic_energy[:n_early])
assert magnetic_energy.max() > 2.0 * early_reference, (
    "Magnetic energy did not grow appreciably over the run: "
    f"max/early = {magnetic_energy.max() / max(early_reference, np.finfo(float).tiny):.3g}"
)

# Check that, at the last snapshot, the total charge density rho equals the sum of the charge
# densities of the two electron beams and the ions.
rho_1, _ = ts.get_field(field="rho_electrons_1", iteration=iterations[-1], plot=False)
rho_2, _ = ts.get_field(field="rho_electrons_2", iteration=iterations[-1], plot=False)
rho_i, _ = ts.get_field(field="rho_ions", iteration=iterations[-1], plot=False)
rho, _ = ts.get_field(field="rho", iteration=iterations[-1], plot=False)

assert np.all(np.isfinite(rho_1))
assert np.all(np.isfinite(rho_2))
assert np.all(np.isfinite(rho_i))
assert np.all(np.isfinite(rho))
# Typical size of the charge density; sets the absolute tolerance of the comparison below.
charge_scale = abs(q_e) * n_0
np.testing.assert_allclose(
    rho, rho_1 + rho_2 + rho_i, rtol=1.0e-7, atol=1.0e-10 * charge_scale
)

# At the end of the run, B_z must be the largest magnetic-field component.
# - B_z > B_y is expected from the physics: the beams move along y, which does not produce B_y.
# - B_z > B_x comes from the grid, not from the physics: both B_x and B_z grow, but the finer
#   spacing in x (dx < dz) resolves more of the structures that vary along x, which produce B_z.
# Average over the last 5 snapshots.
late = slice(max(1, len(t) - 5), len(t))
Bx_late = np.mean(Bx_2[late])
By_late = np.mean(By_2[late])
Bz_late = np.mean(Bz_2[late])
dominance_factor = 1.0
assert Bz_late > dominance_factor * max(Bx_late, By_late), (
    "Expected late-time B_z^2 to be the leading magnetic component: "
    f"B_z^2={Bz_late:.6e}, B_x^2={Bx_late:.6e}, B_y^2={By_late:.6e}"
)

# Choose the time window for the growth-rate fit. The run is deliberately short: it stops while
# B_z is still growing exponentially, before the instability saturates (stops growing).
# B_z^2 therefore first goes through a short start-up phase, while the fastest-growing structures
# emerge from the particle noise, and then grows exponentially until the last snapshot. We skip
# the start-up phase and fit everything after it.
#
# Exponential growth is a straight line on a log scale, so work with ln(B_z^2). Snapshots with
# B_z^2 == 0 give ln = -inf; they are left out below (the warning is silenced).
with np.errstate(divide="ignore", invalid="ignore"):
    ln_Bz_2 = np.log(Bz_2)

# Snapshots where B_z^2 > 0, i.e., where the logarithm is a regular number.
finite_positive = np.flatnonzero(np.isfinite(ln_Bz_2))
assert finite_positive.size >= MIN_FIT_POINTS, (
    f"Need at least {MIN_FIT_POINTS} positive B_z^2 diagnostics to fit growth"
)

# Initial noise level: average of ln(B_z^2) over the first few snapshots. Only used in the
# printed output and in error messages.
noise_sample_count = min(5, max(2, finite_positive.size // 10))
noise_indices = finite_positive[:noise_sample_count]
noise_floor = np.mean(ln_Bz_2[noise_indices])
# How much ln(B_z^2) rises in total, from the noise level to its maximum.
dynamic_range = np.max(ln_Bz_2[finite_positive]) - noise_floor

# During the start-up phase the growth rate is still changing, so there is no sharp feature in
# the data that marks its end. Instead, the fit starts at a fixed physical time
# (omega_p * t >= FIT_START_OMEGA_P_T). This also keeps the window the same if the snapshots are
# written more or less often. The fit then runs until the last snapshot.
candidate = np.flatnonzero(np.isfinite(ln_Bz_2) & (omega_p * t >= FIT_START_OMEGA_P_T))
assert candidate.size >= MIN_FIT_POINTS, (
    "B_z^2 did not grow past the initial transient. "
    f"t_final={t[-1]:.6e} s, ln-noise floor={noise_floor:.6g}, "
    f"ln(B_z^2) dynamic range={dynamic_range:.3g}"
)

# Use one uninterrupted window (snapshots with B_z^2 == 0 would leave gaps).
fit_indices = largest_contiguous_block(candidate)
# The window must contain enough snapshots and show a clear rise of B_z^2.
fit_ln_Bz_2_rise = ln_Bz_2[fit_indices[-1]] - ln_Bz_2[fit_indices[0]]
assert (
    fit_indices.size >= MIN_FIT_POINTS and fit_ln_Bz_2_rise >= MIN_FIT_LN_BZ_2_RISE
), (
    "The B_z^2 growth window after the initial transient is too short: "
    f"{fit_indices.size} diagnostics and {fit_ln_Bz_2_rise:.3g} rise in ln(B_z^2) "
    f"between t={t[fit_indices[0]]:.6e} s and t={t[fit_indices[-1]]:.6e} s."
)

fit_t = t[fit_indices]
fit_ln_Bz_2 = ln_Bz_2[fit_indices]
slope, intercept, r_sq = linear_fit(fit_t, fit_ln_Bz_2)
# B_z grows as exp(gamma * t), so B_z^2 grows as exp(2 * gamma * t) and ln(B_z^2) has slope
# 2 * gamma. The growth rate is therefore half the fitted slope.
gamma_fit = 0.5 * slope

# For the plot: draw a line with the theoretical slope, shifted up or down so that it passes
# through the data in the fit window. This makes it easy to compare the two slopes by eye.
theory_intercept = np.mean(fit_ln_Bz_2 - 2.0 * gamma_theory * fit_t)
# Theory and fit lines, drawn over the whole run (not just the fit window).
theory_ln = theory_intercept + 2.0 * gamma_theory * t
fit_ln = intercept + slope * t

print(f"Fit window: {fit_t[0]:.6e} s to {fit_t[-1]:.6e} s")
print(f"Fit points: {fit_indices.size}")
print(f"ln(B_z^2) dynamic range: {dynamic_range:.6g}")
print(f"ln(B_z^2) rise over fit window: {fit_ln_Bz_2_rise:.6g}")
print(f"Fit R^2: {r_sq:.6f}")
print(f"Fitted gamma: {gamma_fit:.6e} 1/s")
print(f"gamma_fit / gamma_theory: {gamma_fit / gamma_theory:.6f}")

# Plot, on a log scale, the sums of B_x^2, B_y^2, and B_z^2 over time, together with the
# theoretical and fitted growth.
fig, ax = plt.subplots()
ax.semilogy(t, Bx_2, label=r"$\sum B_x^2$")
ax.semilogy(t, By_2, label=r"$\sum B_y^2$")
ax.semilogy(t, Bz_2, label=r"$\sum B_z^2$")
ax.semilogy(
    t,
    np.exp(theory_ln),
    "--",
    label=r"$\propto e^{2 \beta \omega_p t}$",
)
ax.semilogy(t, np.exp(fit_ln), ":", label=r"fit to $\sum B_z^2$")
# Shade the fit window in gray.
ax.axvspan(fit_t[0], fit_t[-1], color="0.85", label="fit window")

# Set the vertical range from the smallest and largest plotted values. Zeros are left out, since
# they cannot be shown on a log scale.
positive_plot_values = np.concatenate(
    (
        Bx_2[Bx_2 > 0.0],
        By_2[By_2 > 0.0],
        Bz_2[Bz_2 > 0.0],
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

# Make animations of B_z and of the charge density of the electrons_1 beam. For the charge
# density, white is the initial average value, so the filaments formed by the instability show
# up as red and blue regions (above and below that average).
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

# Check the fitted growth rate only now, after plotting, so that the plots exist even if the
# check fails. The rate must be a positive number, the data must lie close to a straight line
# (R^2 > 0.8), and the rate must be close to the theoretical value.
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
