#!/usr/bin/env python3

"""
This test checks the angular distribution of electron-positron pairs
produced by linear Breit-Wheeler pair production (gamma gamma -> e+ e-).

Two monoenergetic photon populations counter-propagate head-on along the
x-axis. The test is run at normalized momenta ux = +/-2.8 and +/-1.e6
(in units of m_e*c). Since the total momentum is zero, the
center-of-momentum (CM) frame coincides with the lab frame.

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

import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, m_e

sys.path.append("../../../Tools/Parser/")
from input_file_parser import parse_input_file


def load_reduced_diagnostic(filename):
    """Load a reduced diagnostic as a dict mapping header names to columns."""
    with open(filename) as diagnostic_file:
        names = re.findall(r"\[\d+\]([^\[\s]+)", diagnostic_file.readline())
    data = np.loadtxt(filename)
    data = np.atleast_2d(data)
    assert len(names) == data.shape[1], f"{filename}: header/data column mismatch"
    return dict(zip(names, data.T))


def get_reduced_column(data, name):
    """Return the data column for an exact reduced-diagnostic column name."""
    try:
        return data[name]
    except KeyError as error:
        available_names = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Missing reduced diagnostic column {name!r}; available columns: "
            f"{available_names}"
        ) from error


def lbw_integral_transformed(beta, one_minus_beta2, angular_rapidity):
    """Integral of the BW cross section for y = atanh(beta*cos(theta))."""
    tanh_y = np.tanh(angular_rapidity)
    sech2_y = 1.0 / np.cosh(angular_rapidity) ** 2
    cos_theta = tanh_y / beta
    coefficient = 2.0 + 2.0 * one_minus_beta2 - one_minus_beta2**2
    return (
        -(cos_theta * one_minus_beta2**2) / sech2_y
        - cos_theta
        + coefficient * angular_rapidity / beta
    )


def check_energy_conservation(ekin_data, num_data):
    """Check total energy conservation using reduced diagnostics."""
    ekin_photonA = get_reduced_column(ekin_data, "photonA(J)")
    ekin_photonB = get_reduced_column(ekin_data, "photonB(J)")
    ekin_electron = get_reduced_column(ekin_data, "electron(J)")
    ekin_positron = get_reduced_column(ekin_data, "positron(J)")
    num_phys_electron = get_reduced_column(num_data, "electron_weight()")
    num_phys_positron = get_reduced_column(num_data, "positron_weight()")
    total_energy = (
        ekin_photonA
        + ekin_photonB
        + ekin_electron
        + ekin_positron
        + m_e * c**2 * (num_phys_electron + num_phys_positron)
    )
    max_de = np.max(np.abs(total_energy - total_energy[0]))
    assert np.allclose(total_energy, total_energy[0], rtol=5e-10, atol=0.0), (
        f"Energy conservation failed: max |E - E(0)| = {max_de:.3e} J"
    )


def check_momentum_conservation(mom_data, momentum_abs_tol):
    """Check total momentum conservation using reduced diagnostics."""
    # In this setup, total momentum is expected to remain zero.

    for label in "xyz":
        p = get_reduced_column(mom_data, f"total_{label}(kg*m/s)")
        assert np.abs(p[0]) < momentum_abs_tol, (
            f"Initial total p{label}={p[0]:.3e} exceeds {momentum_abs_tol:.3e}"
        )

        max_dp = np.max(np.abs(p - p[0]))
        assert max_dp < momentum_abs_tol, (
            f"p{label} conservation failed: "
            f"max |p{label} - p{label}(0)| = {max_dp:.3e} "
            f"exceeds {momentum_abs_tol:.3e}"
        )


def check_angular_distribution(ts, ux_photon):
    """Check angular distribution of produced pairs against BW theory."""
    # For head-on collision of equal-energy photons:
    # s = ux_photon^2 and beta = sqrt(1 - 1/s)
    s = ux_photon**2
    one_minus_beta2 = 1.0 / s
    max_angular_rapidity = np.arccosh(ux_photon)
    beta = np.tanh(max_angular_rapidity)
    gamma_expected = ux_photon

    it = ts.iterations[-1]

    ux_e, uy_e, uz_e, w_e = ts.get_particle(
        ["ux", "uy", "uz", "w"], species="electron", iteration=it
    )

    # ux/uy/uz are normalized momenta (gamma*beta) for electrons
    u_mag = np.sqrt(ux_e**2 + uy_e**2 + uz_e**2)
    gamma = np.sqrt(1.0 + u_mag**2)
    gamma_matches = np.all(np.isclose(gamma, gamma_expected, rtol=1e-5))

    # cos(theta) in the CM frame (= lab frame for this setup).
    # The BW polar angle is measured from the collision axis, which is
    # the direction of photon 1 (along +x in this test).
    cos_theta = ux_e / u_mag

    # The transformed coordinate remains well resolved when the distribution is
    # strongly peaked near cos(theta) = +/-1.
    angular_rapidity = np.arctanh(np.clip(beta * cos_theta, -beta, beta))
    integral_at_one = lbw_integral_transformed(
        beta, one_minus_beta2, max_angular_rapidity
    )
    cdf = (
        0.5
        + 0.5
        * lbw_integral_transformed(beta, one_minus_beta2, angular_rapidity)
        / integral_at_one
    )

    # Inverse-transform sampling makes the theoretical CDF values uniform.
    order = np.argsort(cdf)
    sorted_cdf = cdf[order]
    normalized_weights = w_e[order] / np.sum(w_e)
    empirical_cdf = np.cumsum(normalized_weights)
    empirical_cdf_before = empirical_cdf - normalized_weights
    ks_distance = max(
        np.max(np.abs(empirical_cdf - sorted_cdf)),
        np.max(np.abs(empirical_cdf_before - sorted_cdf)),
    )
    print(f"Weighted CDF distance: {ks_distance:.6f}")

    # Binned histogram in angular rapidity, weighted by particle weight.
    n_bins = 20
    bins = np.linspace(-max_angular_rapidity, max_angular_rapidity, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    hist, _ = np.histogram(angular_rapidity, bins=bins, weights=w_e)

    # Expected bin counts from exact differences of the analytical CDF.
    cdf_edges = (
        0.5
        + 0.5 * lbw_integral_transformed(beta, one_minus_beta2, bins) / integral_at_one
    )
    expected_counts = hist.sum() * np.diff(cdf_edges)

    # Verify the transformed integral against the total cross-section expression.
    term1 = (
        (2.0 + 2.0 * one_minus_beta2 - one_minus_beta2**2) * 2.0 * max_angular_rapidity
    )
    term2 = -2.0 * beta * (1.0 + one_minus_beta2)
    integral_f_analytical = (term1 + term2) / beta
    integral_matches = np.isclose(
        2.0 * integral_at_one, integral_f_analytical, rtol=1e-6
    )

    # Compare the binned simulation result with theory.
    rel_err = np.abs(hist - expected_counts) / expected_counts
    max_rel_err = np.max(rel_err)
    print(f"Max relative error in angular distribution: {max_rel_err:.4f}")

    y_fine = np.linspace(-max_angular_rapidity, max_angular_rapidity, 500)
    cdf_fine = (
        0.5
        + 0.5
        * lbw_integral_transformed(beta, one_minus_beta2, y_fine)
        / integral_at_one
    )
    theory_fine = hist.sum() * bin_width * np.gradient(cdf_fine, y_fine)

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist, width=bin_width * 0.9, alpha=0.6, label="Simulation")
    ax.plot(y_fine, theory_fine, "r-", lw=2, label="BW theory")
    ax.set_xlabel(r"$\operatorname{atanh}(\beta\cos\theta^*)$")
    ax.set_ylabel("Weighted pair count")
    ax.set_title(
        rf"BW angular distribution ($u_\gamma = {ux_photon}$, "
        rf"$\beta = {beta:.3f}$)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("angular_distribution.png", dpi=150)
    print("Saved angular_distribution.png")

    assert gamma_matches, (
        f"Expected gamma={gamma_expected}, "
        f"got min={gamma.min():.6f}, max={gamma.max():.6f}"
    )
    assert ks_distance < 0.01, (
        f"Angular distribution does not match BW theory: KS distance={ks_distance:.6f}"
    )
    assert integral_matches, (
        f"Transformed integral {2.0 * integral_at_one:.8f} != "
        f"analytical integral {integral_f_analytical:.8f}"
    )
    assert max_rel_err < 0.15, (
        f"Angular distribution does not match BW theory: max_rel_err={max_rel_err:.4f}"
    )


def main():
    ts = OpenPMDTimeSeries("diags/diag1/")
    ekin_data = load_reduced_diagnostic("diags/reducedfiles/ParticleEnergy.txt")
    num_data = load_reduced_diagnostic("diags/reducedfiles/ParticleNumber.txt")
    mom_data = load_reduced_diagnostic("diags/reducedfiles/ParticleMomentum.txt")

    input_dict = parse_input_file("warpx_used_inputs")
    expected_species = {"photonA", "photonB", "electron", "positron"}
    configured_species = set(input_dict["particles.species_names"])
    assert configured_species == expected_species, (
        "linear Breit-Wheeler angular analysis expects species "
        f"{sorted(expected_species)}, got {sorted(configured_species)}"
    )
    ux_photon_a = float(input_dict["photonA.ux"][0])
    ux_photon_b = float(input_dict["photonB.ux"][0])
    assert np.isclose(ux_photon_b, -ux_photon_a, rtol=0.0, atol=1e-14), (
        "Expected head-on equal-energy photons with photonB.ux = -photonA.ux, "
        f"got photonA.ux={ux_photon_a} and photonB.ux={ux_photon_b}"
    )

    for species in ("photonA", "photonB"):
        for component in ("uy", "uz"):
            value = float(input_dict.get(f"{species}.{component}", [0.0])[0])
            assert np.isclose(value, 0.0, rtol=0.0, atol=1e-14), (
                "Expected zero transverse momentum for head-on collision, "
                f"got {species}.{component}={value}"
            )

    ux_photon = abs(ux_photon_a)
    # Use an absolute tolerance based on a physical momentum scale rather than
    # a relative tolerance against a near-zero reference value (e.g., initial value).
    momentum_abs_tol = 5e-10 * get_reduced_column(ekin_data, "total(J)")[0] / c

    check_energy_conservation(ekin_data, num_data)
    check_momentum_conservation(mom_data, momentum_abs_tol)
    check_angular_distribution(ts, ux_photon)


if __name__ == "__main__":
    main()
