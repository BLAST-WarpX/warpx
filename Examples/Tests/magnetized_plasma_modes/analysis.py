#!/usr/bin/env python3
#
# --- Analysis script for the Darwin/Ohm-solver example producing EM modes.

import argparse

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pywarpx import picmi

constants = picmi.constants

matplotlib.rcParams.update({"font.size": 20})

parser = argparse.ArgumentParser()
solver_group = parser.add_mutually_exclusive_group(required=True)
solver_group.add_argument(
    "--analyze_darwin_sim",
    help="Analyze a simulation run with the Darwin field solver",
    action="store_true",
)
solver_group.add_argument(
    "--analyze_ohm_sim",
    help="Analyze a simulation run with the Ohm (hybrid-PIC) field solver",
    action="store_true",
)
args, left = parser.parse_known_args()
is_darwin = args.analyze_darwin_sim

# load simulation parameters
with open("sim_parameters.dpkl", "rb") as f:
    sim = dill.load(f)

assert sim.solver == ("darwin" if is_darwin else "ohm"), (
    f"--analyze_{'darwin' if is_darwin else 'ohm'}_sim passed but the simulation "
    f"was run with the {sim.solver} solver"
)

if is_darwin:
    if sim.dim == 1:
        field_idx_dict = {"z": 4, "Ez": 7, "Bx": 8, "By": 9}
    else:
        field_idx_dict = {"z": 2, "Ez": 3, "Bx": 4, "By": 5}
else:
    if sim.B_dir == "z":
        field_idx_dict = {"z": 4, "Ez": 7, "Bx": 8, "By": 9}
    else:
        if sim.dim == 1:
            field_idx_dict = {"z": 4, "Ez": 7, "Bx": 8, "By": 9}
        else:
            field_idx_dict = {"z": 2, "Ez": 3, "Bx": 4, "By": 5}

if sim.B_dir == "z":
    data = np.loadtxt("diags/par_field_data.txt", skiprows=1)
else:
    data = np.loadtxt("diags/perp_field_data.txt", skiprows=1)

# step, t, z, Ez, Bx, By = raw_data.T
step = data[:, 0]

num_steps = len(np.unique(step))

# get the spatial resolution
resolution = len(np.where(step == 0)[0]) - 1

# reshape to separate spatial and time coordinates
sim_data = data.reshape((num_steps, resolution + 1, data.shape[1]))

z_grid = sim_data[1, :, field_idx_dict["z"]]
idx = np.argsort(z_grid)[1:]
dz = np.mean(np.diff(z_grid[idx]))
dt = np.mean(np.diff(sim_data[:, 0, 1]))

data = np.zeros((num_steps, resolution, 3))
for i in range(num_steps):
    data[i, :, 0] = sim_data[i, idx, field_idx_dict["Bx"]]
    data[i, :, 1] = sim_data[i, idx, field_idx_dict["By"]]
    data[i, :, 2] = sim_data[i, idx, field_idx_dict["Ez"]]

print(f"Data file contains {num_steps} time snapshots.")
print(f"Spatial resolution is {resolution}")


# Mass ratio m_e / m_i = w_ci / w_ce used in the parallel dispersion relations.
# The Ohm solver treats the electrons as a massless fluid (ratio 0). The Darwin
# solver pushes kinetic electrons with the input mass ratio, which can bring
# the electron cyclotron resonance into the plotted frequency range.
INV_MASS_RATIO = sim.w_ci / sim.w_ce if is_darwin else 0.0


def get_analytic_R_mode(w):
    """Signed wavenumber k l_i of the R (right-hand) mode at frequency w / w_ci.

    With x = |w| / w_ci and r = w_ci / w_ce, the cold-plasma R mode obeys

        (k l_i)^2 = x / (1 - x r) - x / (1 + x)
                  = (1 + r) x^2 / [(1 - x r) (1 + x)] .

    The second form is evaluated because it avoids cancellation at small x.
    For massless electrons (r = 0), it reduces to x^2 / (1 + x).
    Returns nan where the mode does not propagate.
    """
    x = np.abs(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = (1.0 + INV_MASS_RATIO) * x**2 / ((1.0 - x * INV_MASS_RATIO) * (1.0 + x))
    return np.sign(w) * np.sqrt(np.where(k2 > 0.0, k2, np.nan))


def get_analytic_L_mode(w):
    """Signed wavenumber k l_i of the L (left-hand) mode at frequency w / w_ci.

    Same as get_analytic_R_mode, with the two cyclotron resonances swapped:

        (k l_i)^2 = (1 + r) x^2 / [(1 - x) (1 + x r)] .

    For massless electrons (r = 0), it reduces to x^2 / (1 - x).
    Returns nan where the mode does not propagate, i.e., above w_ci.
    """
    x = np.abs(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = (1.0 + INV_MASS_RATIO) * x**2 / ((1.0 - x) * (1.0 + x * INV_MASS_RATIO))
    return np.sign(w) * np.sqrt(np.where(k2 > 0.0, k2, np.nan))


# Settings of the parallel dispersion check.
# Only frequencies above w_ci are tested, where the R mode is the only
# propagating mode.
W_MIN = 1.0
# Bands whose analytic k exceeds this fraction of the largest grid k are
# skipped, because the discrete curl operator distorts the dispersion there.
# This also skips the bands near the electron cyclotron resonance of the Darwin
# runs, where the analytic k diverges.
MAX_K_FRACTION = 0.6
# Bins with power above this fraction of the band's peak power belong to the
# mode and enter the power-weighted centroid of k.
PEAK_FRACTION = 0.25
# Pass/fail tolerances
MAX_MEDIAN_ERROR = 0.10  # maximum median relative error in k
BAND_ERROR = 0.15  # relative error below which a band counts as on the curve
MIN_FRAC_ON_CURVE = 0.75  # minimum fraction of bands on the curve
MIN_SNR = 30.0  # minimum median ratio of peak power to median power in a band


def check_parallel_dispersion(field_kw, k_vals, w):
    """Check that the parallel (B_dir = "z") spectrum follows the R mode.

    For each resolved frequency band above w_ci, measure the wavenumber where
    the spectral power peaks and compare it with get_analytic_R_mode.

    field_kw is the fftshifted spectrum of Bl = (Bx + i By) / sqrt(2), while
    k_vals and w are the plotted axes, in units of 1 / l_i and w_ci.

    Bl keeps only the circular polarization that rotates with the electrons, so
    the R mode appears only at positive raw FFT frequency, for both directions
    of propagation. The figure displays each row of field_kw at minus its raw
    FFT frequency, so the tested R mode appears at negative w on the figure.
    """
    power = np.abs(field_kw) ** 2

    # Recover the raw FFT frequency of each row of field_kw from the plotted w.
    freq = -np.flipud(w)
    k_cut = MAX_K_FRACTION * k_vals.max()

    # Search all positive k except the bins near k = 0, so that the mode must
    # stand out against the whole thermal spectrum, not just a narrow window
    # around the expected k.
    cand = np.where(k_vals > 0.05)[0]
    # The R mode appears at both +k and -k (propagation along and against B).
    # Add the power at -k to the power at +k to double the statistics.
    # On the fftshifted axis, the bin j mirrors the bin 2 * i0 - j, where i0 is
    # the index of k = 0, for both even and odd numbers of grid points.
    power[:, cand] += power[:, 2 * np.argmin(np.abs(k_vals)) - cand]

    rel_errors = []
    peak_snr = []
    print("Parallel R-mode dispersion check:")
    print("     w/w_ci    k l_i (measured)    k l_i (analytic)    rel. err    SNR")
    for i, w in enumerate(freq):
        if w < W_MIN:
            continue
        k_analytic = get_analytic_R_mode(w)
        if not np.isfinite(k_analytic) or k_analytic > k_cut:
            continue
        band_power = power[i][cand]
        peak = band_power.max()
        # Measure k as the power-weighted centroid of the contiguous bins
        # around the peak with power above PEAK_FRACTION of the peak power.
        # The single strongest bin is unreliable: runs with few time snapshots
        # have broad bands with several comparable local maxima. Stopping at
        # the first bin below the threshold on each side keeps separate peaks
        # elsewhere in k out of the centroid.
        above = band_power > PEAK_FRACTION * peak
        lo = hi = np.argmax(band_power)
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        sel = cand[lo : hi + 1]
        k_measured = np.sum(k_vals[sel] * power[i][sel]) / np.sum(power[i][sel])
        rel_errors.append(abs(k_measured - k_analytic) / k_analytic)
        peak_snr.append(peak / np.median(band_power))
        print(
            f"    {w:7.2f}    {k_measured:16.3f}    {k_analytic:16.3f}"
            f"    {rel_errors[-1]:8.3f}    {peak_snr[-1]:7.1f}"
        )

    rel_errors = np.array(rel_errors)
    peak_snr = np.array(peak_snr)
    n_bands = len(rel_errors)
    # Darwin runs with few time snapshots and a low electron cyclotron
    # frequency resolve only a few bands below that resonance.
    assert n_bands >= 2, (
        f"too few resolved frequency bands ({n_bands}) to test dispersion"
    )
    median_error = float(np.median(rel_errors))
    frac_on_curve = float(np.mean(rel_errors < BAND_ERROR))
    median_snr = float(np.median(peak_snr))

    print(f"    resolved frequency bands     : {n_bands}")
    print(
        f"    median relative error in k   : {median_error:.3f} "
        f"(tol {MAX_MEDIAN_ERROR:.2f})"
    )
    print(
        f"    fraction of bands within {BAND_ERROR:.0%} : {frac_on_curve:.2f} "
        f"(tol {MIN_FRAC_ON_CURVE:.2f})"
    )
    print(f"    median peak signal-to-noise  : {median_snr:.1f} (tol {MIN_SNR:g})")

    assert median_error < MAX_MEDIAN_ERROR, (
        f"measured spectrum departs from the analytic R-mode dispersion "
        f"(median relative error {median_error:.3f} >= {MAX_MEDIAN_ERROR:.2f})"
    )
    assert frac_on_curve >= MIN_FRAC_ON_CURVE, (
        f"too many frequency bands off the analytic dispersion "
        f"(only {frac_on_curve:.2f} within {BAND_ERROR:.0%})"
    )
    assert median_snr > MIN_SNR, (
        f"spectral peaks are not prominent enough "
        f"(median SNR {median_snr:.1f} <= {MIN_SNR:g}); "
        f"no clear wave mode was excited"
    )


if sim.B_dir == "z":
    global_norm = (
        1.0
        / (2.0 * constants.mu0)
        / ((3.0 / 2) * sim.n_plasma * sim.T_plasma * constants.q_e)
    )
else:
    global_norm = (
        constants.ep0 / 2.0 / ((3.0 / 2) * sim.n_plasma * sim.T_plasma * constants.q_e)
    )

if sim.B_dir == "z":
    Bl = (data[:, :, 0] + 1.0j * data[:, :, 1]) / np.sqrt(2.0)
    field_kw = np.fft.fftshift(np.fft.fft2(Bl))
else:
    field_kw = np.fft.fftshift(np.fft.fft2(data[:, :, 2]))

w_norm = sim.w_ci
if sim.B_dir == "z":
    k_norm = 1.0 / sim.l_i
else:
    k_norm = 1.0 / sim.rho_i

k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(resolution, dz)) / k_norm
w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(num_steps, dt)) / w_norm
w = -np.flipud(w)

# aspect = (xmax-xmin)/(ymax-ymin) / aspect_true
extent = [k[0], k[-1], w[0], w[-1]]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 7.25))

if sim.B_dir == "z" and sim.dim == 1:
    if is_darwin:
        vmin = -1 if sim.test else 1.5
        vmax = None if sim.test else 5.0
    else:
        vmin = -3
        vmax = 3.5
else:
    if is_darwin:
        vmin = -2.75
        vmax = 3.25
    else:
        vmin = None
        vmax = None

im = ax1.imshow(
    np.log10(np.abs(field_kw**2) * global_norm),
    extent=extent,
    aspect="equal",
    cmap="inferno",
    vmin=vmin,
    vmax=vmax,
)

# Colorbars
fig.subplots_adjust(right=0.5)
cbar_ax = fig.add_axes([0.525, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax, orientation="vertical")

if sim.B_dir == "z":
    cbar_lab = r"$\log_{10}(\beta_{R/L})$"
else:
    cbar_lab = r"$\log_{10}(\varepsilon_0|E_z|^2/(3n_0k_BT_e))$"
cbar_ax.set_ylabel(cbar_lab, rotation=270, labelpad=30)

if sim.B_dir == "z":
    if is_darwin:
        # finite electron mass: include the electron cyclotron resonance
        L_mode_label = (
            r"$(kl_i)^2=\frac{(1+\Omega_i/\Omega_e)(\omega/\Omega_i)^2}"
            r"{(1-\omega/\Omega_i)(1+\omega/\Omega_e)}$"
        )
        R_mode_label = (
            r"$(kl_i)^2=\frac{(1+\Omega_i/\Omega_e)(\omega/\Omega_i)^2}"
            r"{(1-\omega/\Omega_e)(1+\omega/\Omega_i)}$"
        )
    else:
        L_mode_label = r"$(kl_i)^2=\frac{(\omega/\Omega_i)^2}{1-\omega/\Omega_i}$"
        R_mode_label = r"$(kl_i)^2=\frac{(\omega/\Omega_i)^2}{1+\omega/\Omega_i}$"
    # plot the L mode
    ax1.plot(
        get_analytic_L_mode(w),
        np.abs(w),
        c="limegreen",
        ls="--",
        lw=1.25,
        label="L mode:\n" + L_mode_label,
    )
    # plot the R mode
    ax1.plot(
        get_analytic_R_mode(w),
        -np.abs(w),
        c="limegreen",
        ls="-.",
        lw=1.25,
        label="R mode:\n" + R_mode_label,
    )

    ax1.plot(
        k,
        1.0 + 3.0 * sim.v_ti / w_norm * k * k_norm,
        c="limegreen",
        ls=":",
        lw=1.25,
        label=r"$\omega = \Omega_i + 3v_{th,i} k$",
    )
    ax1.plot(
        k, 1.0 - 3.0 * sim.v_ti / w_norm * k * k_norm, c="limegreen", ls=":", lw=1.25
    )

    if is_darwin:
        # the electron cyclotron branch only exists with the Darwin solver,
        # which treats electrons kinetically rather than as a fluid
        ax1.plot(
            k,
            -sim.w_ce / sim.w_ci
            - k
            * k_norm
            / w_norm
            * 3.0
            * np.sqrt(sim.T_plasma * constants.q_e / constants.m_e),
            c="pink",
            ls="-.",
            lw=1.25,
            label=r"$\omega = \Omega_{e} + 3v_{th,e} k$",
        )
        ax1.plot(
            k,
            -sim.w_ce / sim.w_ci
            + k
            * k_norm
            / w_norm
            * 3.0
            * np.sqrt(sim.T_plasma * constants.q_e / constants.m_e),
            c="pink",
            ls="-.",
            lw=1.25,
        )

else:
    if is_darwin:
        ax1.plot(
            k,
            k * k_norm * sim.vA / w_norm,
            c="limegreen",
            ls="-.",
            lw=1.5,
            label=r"$\omega = v_Ak$",
        )

        w_pi_SI = sim.w_pi * sim.w_pe_SI / sim.w_pe
        w_LH = 1.0 / np.sqrt(1.0 / (sim.w_ci * sim.w_ce) + 1.0 / w_pi_SI**2)
        ax1.axhline(w_LH / w_norm, ls="--", c="pink", label=r"$\omega_{LH}$")

    else:
        # digitized values from Munoz et al. (2018)
        x = [
            0.006781609195402272,
            0.1321379310344828,
            0.2671034482758621,
            0.3743678160919539,
            0.49689655172413794,
            0.6143908045977011,
            0.766022988505747,
            0.885448275862069,
            1.0321149425287355,
            1.193862068965517,
            1.4417701149425288,
            1.7736781609195402,
        ]
        y = [
            -0.033194664836814436,
            0.5306857657503109,
            1.100227301968521,
            1.5713856842646996,
            2.135780760818287,
            2.675601492473303,
            3.3477291246729854,
            3.8469357121413563,
            4.4317021915340735,
            5.1079898786293265,
            6.10275764463696,
            7.310074194793499,
        ]
        ax1.plot(x, y, c="limegreen", ls="-.", lw=1.5, label="X mode")

        x = [
            3.953609195402299,
            3.7670114942528734,
            3.5917471264367817,
            3.39735632183908,
            3.1724137931034484,
            2.9408045977011494,
            2.685977011494253,
            2.4593563218390804,
            2.2203218390804595,
            2.0158850574712646,
            1.834183908045977,
            1.6522758620689655,
            1.4937471264367814,
            1.3427586206896551,
            1.2075402298850575,
        ]
        y = [
            4.427971008277223,
            4.458335120298495,
            4.481579963117039,
            4.495861388686366,
            4.544581206844791,
            4.587425483552773,
            4.638160998413175,
            4.698631899472488,
            4.757987734271133,
            4.813955483123902,
            4.862332203971352,
            4.892481880173264,
            4.9247759145687695,
            4.947934983059571,
            4.953124329888064,
        ]
        ax1.plot(x, y, c="limegreen", ls=":", lw=2)

    x = [
        3.9732873563218387,
        3.6515862068965514,
        3.306275862068966,
        2.895655172413793,
        2.4318850574712645,
        2.0747586206896553,
        1.8520229885057473,
        1.6589195402298849,
        1.4594942528735633,
        1.2911724137931033,
        1.1551264367816092,
        1.0335402298850576,
        0.8961149425287356,
        0.7419770114942528,
        0.6141379310344828,
        0.4913103448275862,
    ]
    y = [
        1.1145945018655916,
        1.1193978642192393,
        1.1391259596002916,
        1.162971222713042,
        1.1986533430544237,
        1.230389844319595,
        1.2649997855641806,
        1.3265857528841618,
        1.3706737573444268,
        1.4368486511986962,
        1.4933310460179268,
        1.5485268259210019,
        1.6386327572157655,
        1.7062658146416778,
        1.7828194021529358,
        1.8533687867221342,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2, label="Bernstein modes")

    x = [
        3.9669885057471266,
        3.6533333333333333,
        3.3213563218390805,
        2.9646896551724136,
        2.6106436781609195,
        2.2797011494252875,
        1.910919540229885,
        1.6811724137931034,
        1.4499540229885057,
        1.2577011494252872,
        1.081057471264368,
        0.8791494252873564,
        0.7153103448275862,
    ]
    y = [
        2.2274306300124374,
        2.2428271218424327,
        2.272505039241755,
        2.3084873697302397,
        2.3586224642964364,
        2.402667581592829,
        2.513873997512545,
        2.5859673199811297,
        2.6586610627439207,
        2.7352146502551786,
        2.8161427284813656,
        2.887850066475104,
        2.9455761890466183,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2)

    x = [
        3.9764137931034487,
        3.702022988505747,
        3.459793103448276,
        3.166712643678161,
        2.8715862068965516,
        2.5285057471264367,
        2.2068505747126435,
        1.9037011494252871,
        1.6009885057471265,
        1.3447816091954023,
        1.1538850574712645,
        0.9490114942528736,
    ]
    y = [
        3.3231976669382854,
        3.34875841660591,
        3.378865205643951,
        3.424454260839731,
        3.474160483767209,
        3.522194107303684,
        3.6205343740618434,
        3.7040356821203417,
        3.785435519149119,
        3.868851052879873,
        3.9169704507440923,
        3.952481022429987,
    ]
    ax1.plot(x, y, c="limegreen", ls=":", lw=2)

# ax1.legend(loc='upper left')
# The longer Darwin R/L labels overlap the colorbar label at the default size.
fig.legend(loc=7, fontsize=14 if (is_darwin and sim.B_dir == "z") else 18)

if sim.B_dir == "z":
    ax1.set_xlabel(r"$k l_i$")
    ax1.set_title(r"$B_{R/L} = B_x \pm iB_y$")
    fig.suptitle("Parallel EM modes")
    if is_darwin:
        ax1.set_xlim(-4.5, 4.5)
        ax1.set_ylim(-12, 3)
    else:
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-6, 3)
    dir_str = "par"
else:
    ax1.set_xlabel(r"$k \rho_i$")
    ax1.set_title(r"$E_z(k, \omega)$")
    fig.suptitle(f"Perpendicular EM modes (ion Bernstein) - {sim.dim}D")
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 8)
    dir_str = "perp"

ax1.set_ylabel(r"$\omega / \Omega_i$")

if is_darwin:
    plt.savefig(
        f"spectrum_{dir_str}_{sim.dim}d_{sim.C_SI}_C_SI.png",
        bbox_inches="tight",
    )
else:
    plt.savefig(
        f"spectrum_{dir_str}_{sim.dim}d_{sim.substeps}_substeps_{sim.eta}_eta.png",
        bbox_inches="tight",
    )
if not sim.test:
    plt.show()

# Automated check of the parallel modes. The perpendicular (Bernstein) modes
# are only compared visually with the digitized reference data above.
if sim.B_dir == "z":
    check_parallel_dispersion(field_kw, k, w)
