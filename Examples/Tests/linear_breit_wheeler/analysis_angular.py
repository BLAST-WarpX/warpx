#!/usr/bin/env python3

"""
This test checks the angular distribution of electron-positron pairs
produced by linear Breit-Wheeler pair production (gamma gamma -> e+ e-).

Two monoenergetic photon populations counter-propagate head-on along the
x-axis with normalized momentum ux = +/-2.8 (in units of m_e*c).
Since the total momentum is zero, the center-of-momentum (CM) frame
coincides with the lab frame.

The test verifies:
  1. Energy and momentum conservation (via reduced diagnostics).
  2. Correct lepton Lorentz factor (kinematics: gamma_lepton = u_photon).
  3. The angular distribution of the produced pairs matches the
     Breit-Wheeler differential cross section in the CM frame:
        dsigma/dx ~ [1 + 2*beta^2 - 2*beta^4
                       - 2*beta^2*(1 - beta^2)*x^2
                       - beta^4*x^4] / (1 - beta^2*x^2)^2
        where
        - x = cos(theta), theta the polar angle of the outgoing leptons
        - beta = sqrt(1 - 1/s) is the lepton velocity in the CM frame
        - s = (photon_energy/(m_e*c^2)) ^2
"""

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, m_e

do_plot = True


def lbw_diff_xsec(beta, cos_theta):
    """BW differential cross section dsigma/d(cos(theta))."""
    b2 = beta**2
    b4 = b2**2
    x2 = cos_theta**2
    num = 1.0 + 2.0 * b2 - 2.0 * b4 - 2.0 * b2 * (1.0 - b2) * x2 - b4 * x2**2
    den = (1.0 - b2 * x2) ** 2
    return num / den


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
    ux_photon = 2.8  # = photon_energy/(m_e*c^2)

    # For head-on collision of equal-energy photons:
    # s = ux_photon^2 and beta = sqrt(1 - 1/s)
    s = ux_photon**2
    beta = np.sqrt(1.0 - 1.0 / s)
    gamma_expected = ux_photon

    ts = OpenPMDTimeSeries("diags/diag1/")
    it = ts.iterations[-1]

    ux_e, uy_e, uz_e, w_e = ts.get_particle(
        ["ux", "uy", "uz", "w"], species="electron", iteration=it
    )

    # ux/uy/uz are normalized momenta (gamma*beta) for electrons
    u_mag = np.sqrt(ux_e**2 + uy_e**2 + uz_e**2)
    gamma = np.sqrt(1.0 + u_mag**2)
    assert np.all(np.isclose(gamma, gamma_expected, rtol=1e-5)), (
        f"Expected gamma={gamma_expected}, "
        f"got min={gamma.min():.6f}, max={gamma.max():.6f}"
    )

    # cos(theta) in the CM frame (= lab frame for this setup).
    # The BW polar angle is measured from the collision axis, which is
    # the direction of photon 1 (along +x in this test).
    cos_theta = ux_e / u_mag

    # Binned histogram of cos(theta), weighted by particle weight
    n_bins = 20
    bins = np.linspace(-1, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    hist, _ = np.histogram(cos_theta, bins=bins, weights=w_e)

    # Expected bin counts from the unnormalized differential cross section.
    # The total weight fixes the absolute scale; the shape comes from theory.
    theory_shape = lbw_diff_xsec(beta, bin_centers)
    x_fine = np.linspace(-1, 1, 10001)
    integral_f = np.trapezoid(lbw_diff_xsec(beta, x_fine), x_fine)

    # Verify analytical vs numerical integral of f.
    # Analytical: integral of f from -1 to 1 = [(3-b^4)*ln((1+b)/(1-b)) + 2b(b^2-2)] / b
    term1 = (3.0 - beta**4) * np.log((1.0 + beta) / (1.0 - beta))
    term2 = 2.0 * beta * (beta**2 - 2.0)
    integral_f_analytical = (term1 + term2) / beta
    assert np.isclose(integral_f, integral_f_analytical, rtol=1e-6), (
        f"Numerical integral {integral_f:.8f} != analytical {integral_f_analytical:.8f}"
    )

    expected_counts = hist.sum() * bin_width * theory_shape / integral_f

    # Compare simulation with theory (unnormalized)
    rel_err = np.abs(hist - expected_counts) / expected_counts
    max_rel_err = np.max(rel_err)
    print(f"Max relative error in angular distribution: {max_rel_err:.4f}")
    assert np.all(rel_err < 0.15), (
        f"Angular distribution does not match BW theory: max_rel_err={max_rel_err:.4f}"
    )

    if do_plot:
        import matplotlib.pyplot as plt

        cos_fine = np.linspace(-1, 1, 500)
        theory_fine = (
            hist.sum() * bin_width * lbw_diff_xsec(beta, cos_fine) / integral_f
        )

        fig, ax = plt.subplots()
        ax.bar(bin_centers, hist, width=bin_width * 0.9, alpha=0.6, label="Simulation")
        ax.plot(cos_fine, theory_fine, "r-", lw=2, label="BW theory")
        ax.set_xlabel(r"$\cos\theta^*$")
        ax.set_ylabel("Weighted pair count")
        ax.set_title(
            rf"BW angular distribution ($u_\gamma = {ux_photon}$, "
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
