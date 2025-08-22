#!/usr/bin/env python3

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, eV, h, m_e, physical_constants
from scipy.constants import e as q_e

r_e = physical_constants["classical electron radius"][0]  # m

do_plots = True

Q_bunch = 2080.5031144200598 * 30000 * q_e  # Coulomb
gamma_bunch_mean = 30.205798028084185
gamma_bunch_rms = 0.58182474907848347
bunch_sigma_z = 1.0e-6  # meters

laser_energy = 1.0  # Joule
laser_radius = 10.0e-6  # meters
laser_duration = 2.0e-12  # seconds
photon_energy = eV  # Joule
laser_wavelength = h * c / photon_energy  # meters
N_photons = laser_energy / photon_energy

# Check momentum conservation
series = OpenPMDTimeSeries("diags/diag1")
uz, w = series.get_particle(
    ["uz", "w"], species="electron1", iteration=series.iterations[0]
)
pz_init = np.sum(uz * w)

pz_ele1 = []
pz_pho1 = []
px_ele2, py_ele2, pz_ele2 = [], [], []
px_pho2, py_pho2, pz_pho2 = [], [], []

for i, it in enumerate(series.iterations):
    uz, w = series.get_particle(["uz", "w"], species="electron1", iteration=it)
    pz_ele1 = np.append(pz_ele1, (w * uz).sum())

    ux, uy, uz, w = series.get_particle(
        ["ux", "uy", "uz", "w"], species="electron2", iteration=it
    )
    px_ele2 = np.append(px_ele2, (w * ux).sum())
    py_ele2 = np.append(py_ele2, (w * uy).sum())
    pz_ele2 = np.append(pz_ele2, (w * uz).sum())

    ux, uy, uz, w = series.get_particle(
        ["ux", "uy", "uz", "w"], species="photon2", iteration=it
    )
    px_pho2 = np.append(px_pho2, (w * ux).sum() / (m_e * c))
    py_pho2 = np.append(py_pho2, (w * uy).sum() / (m_e * c))
    pz_pho2 = np.append(pz_pho2, (w * uz).sum() / (m_e * c))

assert np.allclose(0.0, px_pho2 + px_ele2, atol=1.0e-9 * abs(pz_init))
assert np.allclose(0.0, py_pho2 + py_ele2, atol=1.0e-9 * abs(pz_init))
assert np.allclose(pz_ele1 + pz_ele2 + pz_pho2, pz_init, atol=1.0e-9 * abs(pz_init))

# Check that the photon fraction is close (within 10%) to
# the estimate, based on the Klein-Nishina formula

# Calculate the expected photon fraction
# - Total Klein-Nishina cross section in electron rest frame:
beta_bunch_mean = np.sqrt(1 - 1.0 / gamma_bunch_mean**2)
photon_p_rest = gamma_bunch_mean * (1 + beta_bunch_mean) * h / laser_wavelength
k = photon_p_rest / (m_e * c)
# For low k, the Klein-Nishina cross-section is essentially the
# Compton cross-section
assert k < 1.0e-3
print(k)
sigma = 8.0 / 3 * np.pi * r_e**2
# - Total number of photons that go through this cross-section
energy_per_surface = laser_energy / (np.pi * laser_radius**2)
nphoton_per_surface = energy_per_surface / (h * c / laser_wavelength)
expected_frac = sigma * nphoton_per_surface

# Get the simulated photon fraction
w2 = series.get_particle(
    ["uz", "w"], species="photon2", iteration=series.iterations[-1]
)

w1 = series.get_particle(
    ["uz", "w"], species="electron1", iteration=series.iterations[0]
)
simulated_frac = np.sum(w2) / (np.sum(w1))

print("Fraction of Compton-scattered photons / bunch electrons")
print(f"From simulation : {simulated_frac}")
print(f"From theory     : {expected_frac}")
print(f"Relative error  : {abs(simulated_frac - expected_frac) / expected_frac:.2%}")
assert abs(simulated_frac - expected_frac) < 0.06 * expected_frac

# Bin the photons on a grid in frequency and angle

freq_min = 0.5
freq_max = 1.2
N_freq = 500
gammatheta_min = 0.0
gammatheta_max = 1.0
N_gammatheta = 100
hist_range = [[freq_min, freq_max], [gammatheta_min, gammatheta_max]]
extent = [freq_min, freq_max, gammatheta_min, gammatheta_max]
fundamental_frequency = 4 * gamma_bunch_mean**2 * c / laser_wavelength

# Compton-scattered photons (in lab frame)
px, py, pz, w = series.get_particle(
    ["ux", "uy", "uz", "w"], species="photon2", iteration=series.iterations[-1]
)
p_perp = np.sqrt(px**2 + py**2)
p_norm = np.sqrt(px**2 + py**2 + pz**2)
photon_scaled_freq = p_norm * c / (h * fundamental_frequency)
gamma_theta = gamma_bunch_mean * np.arctan2(p_perp, -pz)

# Compute histogram
grid, freq_bins, gammatheta_bins = np.histogram2d(
    photon_scaled_freq,
    gamma_theta,
    weights=w,
    range=hist_range,
    bins=[N_freq, N_gammatheta],
)

# Normalize by solid angle, frequency and number of photons
if do_plots:
    import matplotlib.pyplot as plt

    dw = (freq_bins[1] - freq_bins[0]) * 2 * np.pi * fundamental_frequency
    dtheta = (gammatheta_bins[1] - gammatheta_bins[0]) / gamma_bunch_mean
    domega = 2.0 * np.pi * np.sin(gammatheta_bins / gamma_bunch_mean) * dtheta
    grid /= dw * domega[np.newaxis, 1:] * np.sum(w)
    grid = np.where(grid == 0, np.nan, grid)
    plt.imshow(
        grid.T,
        origin="lower",
        extent=extent,
        cmap="gist_earth",
        aspect="auto",
    )
    plt.title(r"Particles, $d^2N/d\omega \,d\Omega$")
    plt.xlabel(r"Scaled energy ($\omega/4\gamma^2\omega_\ell$)")
    plt.ylabel(r"$\gamma \theta$")
    plt.colorbar()
    # Plot theory
    plt.plot(1.0 / (1 + gammatheta_bins**2), gammatheta_bins, color="r")
    plt.show()
    plt.clf()
