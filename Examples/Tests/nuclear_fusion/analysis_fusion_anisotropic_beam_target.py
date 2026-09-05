#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scc
from openpmd_viewer import OpenPMDTimeSeries

sys.path.append(str(Path(__file__).resolve().parents[3] / "Tools" / "Parser"))
from input_file_parser import parse_input_file

MASS = {
    "deuteron": 2.01410177812 * scc.m_u - scc.m_e,
    "triton": 3.0160492779 * scc.m_u - scc.m_e,
    "helion": 3.0160293201 * scc.m_u - 2.0 * scc.m_e,
    "alpha": 4.00260325413 * scc.m_u - 2.0 * scc.m_e,
    "neutron": 1.0013784193052508 * scc.m_p,
}

REACTION_CONFIG = {
    "DD": {
        "e_min": 0.0,
        "e_max": 15.0,
        "output": "deuterium_deuterium_fusion_anisotropic_beam_target_neutron_spectrum.png",
        "xticks": [0.0, 5.0, 10.0, 15.0],
        "scale": 1.0,
        "target_mass": MASS["deuteron"],
        "product_mass": MASS["helion"],
        "species": "neutron_1",
    },
    "DT": {
        "e_min": 10.0,
        "e_max": 30.0,
        "output": "deuterium_tritium_fusion_anisotropic_beam_target_neutron_spectrum.png",
        "xticks": [10.0, 15.0, 20.0, 25.0, 30.0],
        "scale": 2.6,
        "target_mass": MASS["triton"],
        "product_mass": MASS["alpha"],
        "species": "neutron_1",
    },
}

# Reference values obtained with the fixed random seed in the input files. These
# are integral observables, with intentionally broad tolerances for portability
# across compute backends and particle precisions.
REFERENCE_METRICS = {
    "DD": {
        1: {
            "mean_energy": 2.49595,
            "std_energy": 0.214295,
            "mean_cos_theta": -0.002990,
        },
        5: {"mean_energy": 3.62257, "std_energy": 1.39414, "mean_cos_theta": -0.000174},
        10: {
            "mean_energy": 7.13244,
            "std_energy": 3.88327,
            "mean_cos_theta": -0.000100,
        },
    },
    "DT": {
        1: {
            "mean_energy": 14.0822,
            "std_energy": 0.376711,
            "mean_cos_theta": -0.001278,
        },
        5: {"mean_energy": 15.5438, "std_energy": 1.98236, "mean_cos_theta": 0.057142},
        10: {"mean_energy": 19.9284, "std_energy": 4.81147, "mean_cos_theta": 0.088694},
    },
}


def parse_args():
    argv = sys.argv[1:]
    if len(argv) == 2 and argv[1].lower().endswith((".png", ".pdf", ".svg")):
        argv = [argv[0], "--output", argv[1]]

    parser = argparse.ArgumentParser()
    parser.add_argument("diag_dirs", nargs="*")
    parser.add_argument("-o", "--output")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot all supplied diagnostics after validating them",
    )
    args = parser.parse_args(argv)

    if args.output is not None:
        args.plot = True

    return args


def find_used_inputs(diag_dir):
    path = Path(diag_dir)
    for input_path in [
        path / "warpx_used_inputs",
        path.parent / "warpx_used_inputs",
        path.parent.parent / "warpx_used_inputs",
    ]:
        if input_path.exists():
            return input_path
    raise AssertionError(f"Could not find warpx_used_inputs for {diag_dir}.")


def infer_reaction(diag_dir):
    input_dict = parse_input_file(find_used_inputs(diag_dir))
    collision_names = input_dict.get("collisions.collision_names", [])
    inferred_reactions = set()

    for collision_name in collision_names:
        species = input_dict.get(f"{collision_name}.species", [])
        species_types = [
            input_dict.get(f"{species_name}.species_type", [None])[0]
            for species_name in species
        ]
        if species_types.count("deuterium") == 2 and len(species_types) == 2:
            inferred_reactions.add("DD")
        elif (
            species_types.count("deuterium") == 1
            and species_types.count("tritium") == 1
            and len(species_types) == 2
        ):
            inferred_reactions.add("DT")

    assert len(inferred_reactions) == 1, (
        f"Could not infer exactly one supported fusion reaction from "
        f"{find_used_inputs(diag_dir)}; found {sorted(inferred_reactions)}."
    )
    return inferred_reactions.pop()


def infer_beam_momentum(diag_dir):
    input_dict = parse_input_file(find_used_inputs(diag_dir))
    parameter = "deuterium_beam.momentum_function_uz(x,y,z)"
    (beam_momentum,) = input_dict[parameter]
    return float(beam_momentum)


def beam_gamma_beta_percent(diag_dir):
    gamma_beta_percent = 100.0 * infer_beam_momentum(diag_dir)
    rounded_gamma_beta = round(gamma_beta_percent)
    assert np.isclose(gamma_beta_percent, rounded_gamma_beta), (
        f"Unsupported beam momentum u/c = {gamma_beta_percent:g}% in {diag_dir}."
    )
    return rounded_gamma_beta


def infer_label(diag_dir):
    gamma_beta_percent = beam_gamma_beta_percent(diag_dir)
    return rf"$u/c = {gamma_beta_percent:g}\%$"


def weighted_mean(values, weights):
    return np.sum(weights * values) / np.sum(weights)


def get_cm_kinematics(diag_dir, reaction):
    beam_u = infer_beam_momentum(diag_dir)
    beam_gamma = np.sqrt(1.0 + beam_u**2)
    beam_mass = MASS["deuteron"]
    target_mass = REACTION_CONFIG[reaction]["target_mass"]
    product_mass = REACTION_CONFIG[reaction]["product_mass"]

    # The target is stationary and the beam is monoenergetic.  Its input momentum is
    # u/c = gamma*beta, so the total reactant four-momentum gives one exact CM boost
    # for the entire run.
    cm_beta = beam_mass * beam_u / (beam_mass * beam_gamma + target_mass)
    cm_gamma = 1.0 / np.sqrt(1.0 - cm_beta**2)

    # The invariant mass of the reactants fixes the energy and momentum of both
    # products in the CM frame for this two-body reaction.
    cm_mass = np.sqrt(
        beam_mass**2 + target_mass**2 + 2.0 * beam_mass * target_mass * beam_gamma
    )
    neutron_rest_energy = MASS["neutron"] * scc.c**2
    neutron_energy_cm = (
        (cm_mass**2 + MASS["neutron"] ** 2 - product_mass**2)
        * scc.c**2
        / (2.0 * cm_mass)
    )
    neutron_momentum_cm_c = np.sqrt(neutron_energy_cm**2 - neutron_rest_energy**2)
    return cm_beta, cm_gamma, neutron_energy_cm, neutron_momentum_cm_c


def get_neutron_spectrum(diag_dir, reaction):
    ts = OpenPMDTimeSeries(diag_dir)
    iteration = ts.iterations[-1]
    ux, uy, uz, w = ts.get_particle(
        ["ux", "uy", "uz", "w"],
        species=REACTION_CONFIG[reaction]["species"],
        iteration=iteration,
    )

    u2 = ux**2 + uy**2 + uz**2
    gamma = np.sqrt(1.0 + u2)
    m_neutron = MASS["neutron"]
    # Convert neutron momentum u = gamma beta to kinetic energy.
    energy_MeV = (gamma - 1.0) * m_neutron * scc.c**2 / scc.e / 1.0e6
    cm_beta, cm_gamma, neutron_energy_cm, neutron_momentum_cm_c = get_cm_kinematics(
        diag_dir, reaction
    )
    # For this monoenergetic two-body reaction, the lab-frame neutron energy is
    # exactly linear in its CM-frame direction cosine:
    # E_lab = gamma_cm * (E_n_cm + beta_cm * p_n_cm * c * mu_cm).
    neutron_energy_lab = gamma * m_neutron * scc.c**2
    cos_theta_from_energy = (neutron_energy_lab / cm_gamma - neutron_energy_cm) / (
        cm_beta * neutron_momentum_cm_c
    )

    # Independently reconstruct the CM-frame direction from the momentum vector.
    # Longitudinal momentum transforms as
    # p_z,lab*c = gamma_cm*(beta_cm*E_n,cm + p_n,cm*c*cos(theta)),
    # while transverse momentum is invariant under the boost.  These checks catch
    # inconsistencies between the sampled energy/angle and the assigned momentum.
    momentum_c_scale = m_neutron * scc.c**2
    momentum_z_lab_c = uz * momentum_c_scale
    momentum_perp_lab_c = np.sqrt(ux**2 + uy**2) * momentum_c_scale
    cos_theta_from_momentum = (
        momentum_z_lab_c / cm_gamma - cm_beta * neutron_energy_cm
    ) / neutron_momentum_cm_c
    np.testing.assert_allclose(
        cos_theta_from_energy,
        cos_theta_from_momentum,
        rtol=1.0e-9,
        atol=1.0e-9,
        err_msg="CM direction inferred from neutron energy and momentum disagrees",
    )
    np.testing.assert_allclose(
        momentum_perp_lab_c**2,
        neutron_momentum_cm_c**2 * (1.0 - cos_theta_from_momentum**2),
        rtol=1.0e-9,
        atol=1.0e-9 * neutron_momentum_cm_c**2,
        err_msg="Neutron transverse momentum is inconsistent with its CM direction",
    )
    cos_theta = np.clip(cos_theta_from_energy, -1.0, 1.0)
    return energy_MeV, cos_theta, w


def validate_neutron_spectrum(diag_dir, reaction):
    energy_MeV, cos_theta, w = get_neutron_spectrum(diag_dir, reaction)
    assert energy_MeV.size >= 1000, (
        f"Too few neutron macroparticles ({energy_MeV.size}) in {diag_dir}."
    )
    assert np.all(np.isfinite(energy_MeV))
    assert np.all(np.isfinite(cos_theta))
    assert np.all(np.isfinite(w)) and np.all(w > 0.0)

    mean_energy = weighted_mean(energy_MeV, w)
    std_energy = np.sqrt(weighted_mean((energy_MeV - mean_energy) ** 2, w))
    mean_cos_theta = weighted_mean(cos_theta, w)
    metrics = {
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "mean_cos_theta": mean_cos_theta,
    }

    gamma_beta_percent = beam_gamma_beta_percent(diag_dir)
    assert gamma_beta_percent in REFERENCE_METRICS[reaction], (
        f"No reference metrics for u/c = {gamma_beta_percent:g}%."
    )
    reference = REFERENCE_METRICS[reaction][gamma_beta_percent]
    np.testing.assert_allclose(
        [metrics["mean_energy"], metrics["std_energy"]],
        [reference["mean_energy"], reference["std_energy"]],
        rtol=0.05,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["mean_cos_theta"],
        reference["mean_cos_theta"],
        rtol=0.0,
        atol=0.03,
    )
    print(
        f"{reaction}, u/c = {gamma_beta_percent:g}%: "
        f"mean energy = {mean_energy:.6g} MeV, "
        f"energy std. dev. = {std_energy:.6g} MeV, "
        f"mean cos(theta) = {mean_cos_theta:.6g}"
    )


def plot_neutron_spectra(diag_dirs, labels, reaction, config, output):
    e_min = config["e_min"]
    e_max = config["e_max"]
    energy_bins = np.linspace(e_min, e_max, 240)
    # Use the same energy range for the energy-resolved mean emission angle.
    energy_bins_angle = np.linspace(e_min, e_max, 180)
    angle_centers = 0.5 * (energy_bins_angle[:-1] + energy_bins_angle[1:])

    fig, ax_spectrum = plt.subplots(figsize=(6.0, 2.5), constrained_layout=True)
    ax_angle = ax_spectrum.twinx()

    for diag_dir, label in zip(diag_dirs, labels):
        energy_MeV, cos_theta, w = get_neutron_spectrum(diag_dir, reaction)
        assert energy_MeV.size > 0, f"No neutron macroparticles found in {diag_dir}."
        # Convert to degrees only for plotting; arccos is ill-conditioned near mu = +-1.
        theta_deg = np.degrees(np.arccos(cos_theta))

        # Histogram macroparticle weights to obtain the neutron energy spectrum.
        hist, edges = np.histogram(energy_MeV, bins=energy_bins, weights=w)
        assert hist.sum() > 0.0, (
            f"No neutron weight in the plotted energy range for {diag_dir}."
        )
        bin_width = edges[1] - edges[0]
        # Normalize by total weight and bin width so the plotted area is unity.
        spectrum = hist / (hist.sum() * bin_width)
        # Apply the reaction-dependent ad-hoc scaling to the spectrum.
        spectrum *= config.get("scale", 1.0)
        # Plot against bin centers instead of left bin edges.
        centers = 0.5 * (edges[:-1] + edges[1:])

        (line,) = ax_spectrum.plot(centers, spectrum, lw=1.4, label=label)

        # Overlay the weighted mean emission angle in each neutron-energy bin.
        # The first histogram is the numerator, sum(w * theta).
        weighted_angle, _ = np.histogram(
            energy_MeV, bins=energy_bins_angle, weights=w * theta_deg
        )
        # The second histogram is the denominator, sum(w).
        angle_weight, _ = np.histogram(energy_MeV, bins=energy_bins_angle, weights=w)
        valid = angle_weight > 0.0
        mean_angle = np.empty_like(weighted_angle)
        mean_angle[:] = np.nan
        # Leave empty bins as NaN so Matplotlib breaks the dashed curve there.
        mean_angle[valid] = weighted_angle[valid] / angle_weight[valid]
        ax_angle.plot(
            angle_centers,
            mean_angle,
            color=line.get_color(),
            ls="--",
            lw=1.2,
        )

    ax_spectrum.set_xlim(e_min, e_max)
    ax_spectrum.set_ylim(0.0, 2.2)
    ax_angle.set_ylim(0.0, 180.0)
    ax_spectrum.set_xticks(config["xticks"])
    ax_spectrum.set_yticks([0.0, 1.0, 2.0])
    ax_angle.set_yticks([0.0, 60.0, 120.0, 180.0])
    ax_spectrum.set_xlabel(r"$E_n$ (MeV)")
    ax_spectrum.set_ylabel(r"$dN/dE_n$ (arb. units)")
    ax_angle.set_ylabel(r"$\theta_\mathrm{CM}$ (degrees)")
    ax_spectrum.legend(frameon=False, loc="upper right", fontsize=8)
    ax_spectrum.minorticks_on()
    ax_angle.minorticks_on()

    fig.savefig(output, dpi=200)
    print(f"Saved {output}")


def main():
    args = parse_args()
    if not args.diag_dirs:
        args.diag_dirs = [Path("diags/diag1")]

    reactions = {infer_reaction(diag_dir) for diag_dir in args.diag_dirs}
    assert len(reactions) == 1, (
        f"Diagnostics contain different reactions: {sorted(reactions)}. "
        "Analyze them separately."
    )
    reaction = reactions.pop()

    if args.output is None:
        args.output = REACTION_CONFIG[reaction]["output"]

    for diag_dir in args.diag_dirs:
        validate_neutron_spectrum(diag_dir, reaction)

    if args.plot:
        labels = args.labels
        if labels is None:
            labels = [infer_label(diag_dir) for diag_dir in args.diag_dirs]
        assert len(labels) == len(args.diag_dirs), (
            "Number of labels must match diagnostics."
        )

        plot_neutron_spectra(
            args.diag_dirs,
            labels,
            reaction,
            REACTION_CONFIG[reaction],
            args.output,
        )


if __name__ == "__main__":
    main()
