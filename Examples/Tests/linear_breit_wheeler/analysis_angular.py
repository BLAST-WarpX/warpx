#!/usr/bin/env python3

# This test checks the angular distribution of electron-positron pairs
# produced by linear Breit-Wheeler pair production (gamma gamma -> e+ e-).
#
# Two monoenergetic photon populations counter-propagate head-on along the
# x-axis with normalized momentum ux = +/-2.8 (in units of m_e*c).
# Since the total momentum is zero, the center-of-momentum (CM) frame
# coincides with the lab frame.
#
# The test verifies:
#   1. Energy and momentum conservation (via reduced diagnostics).
#   2. Correct lepton Lorentz factor (kinematics: gamma_lepton = u_photon).
#   3. The angular distribution of the produced pairs matches the
#      Breit-Wheeler differential cross section:
#        dsigma/d(cos theta) ~ f(beta, cos theta) with
#        f(beta, x) = [1 + 2*beta^2 - 2*beta^4
#                       - 2*beta^2*(1 - beta^2)*x^2
#                       - beta^4*x^4] / (1 - beta^2*x^2)^2
#      where beta = sqrt(1 - 1/s) is the lepton velocity in the CM frame,
#      s = u_photon^2, and x = cos theta.

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, m_e

do_plot = True


def lbw_diff_xsec(beta, cos_theta):
    """BW differential cross section dsigma/d(cos theta)."""
    b2 = beta**2
    b4 = b2**2
    x2 = cos_theta**2
    num = 1.0 + 2.0 * b2 - 2.0 * b4 - 2.0 * b2 * (1.0 - b2) * x2 - b4 * x2**2
    den = (1.0 - b2 * x2) ** 2
    return num / den


def lbw_pdf(beta, cos_theta):
    """Normalized PDF for cos theta in BW pair production."""
    x = np.linspace(-1, 1, 10001)
    f = lbw_diff_xsec(beta, x)
    norm = np.trapezoid(f, x)
    return lbw_diff_xsec(beta, cos_theta) / norm


def check_energy_conservation():
    """Check total energy conservation using reduced diagnostics."""
    ekin_data = np.loadtxt("diags/reducedfiles/ParticleEnergy.txt")
    ekin_photonA = ekin_data[:, 3]
    ekin_photonB = ekin_data[:, 4]
    ekin_electron = ekin_data[:, 5]
    ekin_positron = ekin_data[:, 6]
    num_data = np.loadtxt("diags/reducedfiles/ParticleNumber.txt")
    num_phys_electron = num_data[:, 10]
    num_phys_positron = num_data[:, 11]
    total_energy = (
        ekin_photonA
        + ekin_photonB
        + ekin_electron
        + ekin_positron
        + m_e * c**2 * (num_phys_electron + num_phys_positron)
    )
    assert np.all(np.isclose(total_energy, total_energy[0], rtol=5e-10, atol=0.0))


def check_momentum_conservation():
    """Check total momentum conservation using reduced diagnostics."""
    mom_data = np.loadtxt("diags/reducedfiles/ParticleMomentum.txt")
    total_px = mom_data[:, 2]
    total_py = mom_data[:, 3]
    total_pz = mom_data[:, 4]
    assert np.all(np.isclose(total_px, total_px[0], rtol=5e-10, atol=0.0))
    assert np.all(np.isclose(total_py, total_py[0], rtol=5e-10, atol=0.0))
    assert np.all(np.isclose(total_pz, total_pz[0], rtol=5e-10, atol=0.0))


def check_angular_distribution():
    """Check angular distribution of produced pairs against BW theory."""
    u_photon = 2.8

    # For head-on collision of equal-energy photons:
    # s = u_photon^2 and beta = sqrt(1 - 1/s)
    s = u_photon**2
    beta = np.sqrt(1.0 - 1.0 / s)
    gamma_expected = u_photon

    ts = OpenPMDTimeSeries("diags/diag1/")
    it = ts.iterations[-1]

    px_e, py_e, pz_e, w_e = ts.get_particle(
        ["ux", "uy", "uz", "w"], species="electron", iteration=it
    )

    # ux/uy/uz are normalized momenta (gamma*beta) for electrons
    p_mag = np.sqrt(px_e**2 + py_e**2 + pz_e**2)
    gamma = np.sqrt(1.0 + p_mag**2)
    assert np.all(np.isclose(gamma, gamma_expected, rtol=1e-5)), (
        f"Expected gamma={gamma_expected}, "
        f"got min={gamma.min():.6f}, max={gamma.max():.6f}"
    )

    # cos(theta*) in the CM frame (= lab frame for this setup).
    # The code assigns the BW polar angle to the z-axis.
    cos_theta = pz_e / p_mag

    # Binned histogram of cos(theta*), weighted by particle weight
    n_bins = 20
    bins = np.linspace(-1, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(cos_theta, bins=bins, weights=w_e)

    # Normalize to a probability density
    bin_width = bins[1] - bins[0]
    hist_norm = hist / (hist.sum() * bin_width)

    # Theoretical PDF
    theory_pdf = lbw_pdf(beta, bin_centers)

    # Compare simulation with theory
    rel_err = np.abs(hist_norm - theory_pdf) / theory_pdf
    max_rel_err = np.max(rel_err)
    print(f"Max relative error in angular distribution: {max_rel_err:.4f}")
    assert np.all(rel_err < 0.15), (
        f"Angular distribution does not match BW theory: max_rel_err={max_rel_err:.4f}"
    )

    if do_plot:
        import matplotlib.pyplot as plt

        cos_fine = np.linspace(-1, 1, 500)
        theory_fine = lbw_pdf(beta, cos_fine)

        fig, ax = plt.subplots()
        ax.bar(
            bin_centers, hist_norm, width=bin_width * 0.9, alpha=0.6, label="Simulation"
        )
        ax.plot(cos_fine, theory_fine, "r-", lw=2, label="BW theory")
        ax.set_xlabel(r"$\cos\theta^*$")
        ax.set_ylabel("Probability density")
        ax.set_title(
            rf"BW angular distribution ($u_\gamma = {u_photon}$, "
            rf"$\beta = {beta:.3f}$)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig("angular_distribution.png", dpi=150)
        print("Saved angular_distribution.png")
        plt.show()


def main():
    check_energy_conservation()
    check_momentum_conservation()
    check_angular_distribution()


if __name__ == "__main__":
    main()
