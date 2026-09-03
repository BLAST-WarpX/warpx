#!/usr/bin/env python3
"""Validate nonlinear finite-volume transport against a latent-energy adiabat.

The fixed-charge model used by the test has, in eV per electron,

    epsilon(T) = T/(gamma-1) + L T^2/(T^2 + Ta^2),

while its pressure remains P = n e T.  The source-free first law therefore
integrates exactly to

    G(T) - G(T0) = ln(n/n0),

where

    G(T) - G(T0) = ln(T/T0)/(gamma-1) + L [F(T) - F(T0)],
    F(T) = T/(T^2 + Ta^2) + atan(T/Ta)/Ta.

The analysis checks this invariant pointwise during a periodic sinusoidal
compression and also requires the sampled nonlinear reference to be visibly
different from the ideal-gas adiabat.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpmd_viewer import OpenPMDTimeSeries

Q_E = 1.602176634e-19
K_B = 1.380649e-23
K_PER_EV = K_B / Q_E


def latent_primitive(temperature_eV, transition_temperature_eV):
    """Return F(T) for the sharpness-two latent-energy shoulder."""
    temperature_eV = np.asarray(temperature_eV, dtype=float)
    ta = float(transition_temperature_eV)
    return (
        temperature_eV / (temperature_eV**2 + ta**2)
        + np.arctan(temperature_eV / ta) / ta
    )


def g_delta(
    temperature_eV,
    *,
    initial_temperature_eV,
    transition_temperature_eV,
    latent_energy_eV,
    gamma,
):
    """Return G(T)-G(T0), normalized to vanish at the initial state."""
    temperature_eV = np.asarray(temperature_eV, dtype=float)
    t0 = float(initial_temperature_eV)
    return np.log(temperature_eV / t0) / (gamma - 1.0) + latent_energy_eV * (
        latent_primitive(temperature_eV, transition_temperature_eV)
        - latent_primitive(t0, transition_temperature_eV)
    )


def nonlinear_temperature(density_ratio, args):
    """Invert the monotone analytic adiabat with a vectorized bisection."""
    density_ratio = np.asarray(density_ratio, dtype=float)
    target = np.log(density_ratio)
    lo = np.full_like(target, args.initial_temperature_eV * 1.0e-3)
    hi = np.full_like(target, args.initial_temperature_eV * 1.0e3)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = (
            g_delta(
                mid,
                initial_temperature_eV=args.initial_temperature_eV,
                transition_temperature_eV=args.transition_temperature_eV,
                latent_energy_eV=args.latent_energy_eV,
                gamma=args.gamma,
            )
            < target
        )
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--diag-dir", default="diags/field_diags")
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--initial-temperature-eV", type=float, default=100.0)
    parser.add_argument("--transition-temperature-eV", type=float, default=100.0)
    parser.add_argument("--latent-energy-eV", type=float, default=1000.0)
    parser.add_argument("--n0", type=float, default=2.0e20)
    parser.add_argument("--n-floor-frac", type=float, default=0.05)
    parser.add_argument(
        "--significant-density-swing",
        type=float,
        default=0.03,
        help="score cells this far from n/n0=1 (iteration zero is excluded)",
    )
    parser.add_argument(
        "--min-density-swing",
        type=float,
        default=0.08,
        help="minimum peak |n/n0-1| required to make the test discriminating",
    )
    parser.add_argument(
        "--tol-median-invariant",
        type=float,
        default=0.03,
        help="allowed median absolute error in G(T)-G(T0)-ln(n/n0)",
    )
    parser.add_argument(
        "--tol-max-invariant",
        type=float,
        default=0.15,
        help="allowed maximum absolute error in the nonlinear invariant",
    )
    parser.add_argument(
        "--min-ideal-separation",
        type=float,
        default=0.03,
        help="required maximum relative separation of nonlinear and ideal references",
    )
    parser.add_argument(
        "--max-error-ratio-to-ideal",
        type=float,
        default=0.5,
        help="nonlinear median T error must be below this fraction of ideal error",
    )
    parser.add_argument("--out", default="latent_adiabat_check.png")
    args = parser.parse_args(argv)

    series = OpenPMDTimeSeries(args.diag_dir)
    iterations = list(series.iterations)
    times = np.asarray(series.t, dtype=float)
    if len(iterations) < 2:
        raise SystemExit(f"Need at least two dumps in {args.diag_dir}")

    def z_average(name, iteration):
        array, info = series.get_field(name, iteration=iteration)
        return np.asarray(info.x, dtype=float), np.asarray(array, dtype=float).mean(
            axis=0
        )

    coordinate = None
    temperature, density = [], []
    for iteration in iterations:
        x, te = z_average("Te", iteration)
        _, rho = z_average("rho", iteration)
        if coordinate is None:
            coordinate = x
        temperature.append(te * K_PER_EV)
        density.append(rho / Q_E)
    temperature = np.asarray(temperature)
    density = np.asarray(density)
    density_ratio = density / args.n0

    reference_temperature = nonlinear_temperature(density_ratio, args)
    ideal_temperature = args.initial_temperature_eV * density_ratio ** (
        args.gamma - 1.0
    )
    invariant_residual = g_delta(
        temperature,
        initial_temperature_eV=args.initial_temperature_eV,
        transition_temperature_eV=args.transition_temperature_eV,
        latent_energy_eV=args.latent_energy_eV,
        gamma=args.gamma,
    ) - np.log(density_ratio)

    valid = density_ratio > args.n_floor_frac
    evolved = np.broadcast_to(np.arange(len(iterations))[:, None] > 0, density.shape)
    significant = (
        valid & evolved & (np.abs(density_ratio - 1.0) > args.significant_density_swing)
    )
    if not np.any(significant):
        raise SystemExit("No evolved cells have a significant density perturbation")

    abs_invariant_error = np.abs(invariant_residual[significant])
    median_invariant_error = float(np.median(abs_invariant_error))
    max_invariant_error = float(np.max(abs_invariant_error))
    nonlinear_temperature_error = np.abs(
        temperature - reference_temperature
    ) / np.maximum(reference_temperature, 1.0e-30)
    ideal_temperature_error = np.abs(temperature - ideal_temperature) / np.maximum(
        ideal_temperature, 1.0e-30
    )
    median_nonlinear_temperature_error = float(
        np.median(nonlinear_temperature_error[significant])
    )
    median_ideal_temperature_error = float(
        np.median(ideal_temperature_error[significant])
    )
    error_ratio_to_ideal = median_nonlinear_temperature_error / max(
        median_ideal_temperature_error, 1.0e-30
    )
    ideal_separation = np.abs(reference_temperature - ideal_temperature) / np.maximum(
        reference_temperature, 1.0e-30
    )
    max_ideal_separation = float(np.max(ideal_separation[significant]))
    peak_density_swing = float(np.max(np.abs(density_ratio[valid] - 1.0)))

    print("=" * 72)
    print("Fixed-charge latent-energy nonlinear adiabat")
    print(
        f"  gamma={args.gamma:.6g}, T0=Ta={args.initial_temperature_eV:.6g} eV, "
        f"L={args.latent_energy_eV:.6g} eV/electron"
    )
    print(f"  peak density swing: {peak_density_swing:.2%}")
    print(
        "  |G(T)-G(T0)-ln(n/n0)|: "
        f"median={median_invariant_error:.6e}, max={max_invariant_error:.6e}"
    )
    print(
        "  median relative T error: "
        f"nonlinear={median_nonlinear_temperature_error:.3%}, "
        f"ideal={median_ideal_temperature_error:.3%}, "
        f"ratio={error_ratio_to_ideal:.3f}"
    )
    print(f"  maximum analytic nonlinear/ideal separation: {max_ideal_separation:.3%}")
    print("=" * 72)

    fig, (profile_axis, invariant_axis) = plt.subplots(1, 2, figsize=(13, 5))
    time_indices = sorted(
        set(np.linspace(0, len(iterations) - 1, 5, dtype=int).tolist())
    )
    for index in time_indices:
        color = plt.cm.viridis(index / max(len(iterations) - 1, 1))
        label = f"t={times[index] * 1.0e6:.2f}" + r" $\mu$s"
        profile_axis.plot(
            coordinate * 100.0,
            temperature[index],
            color=color,
            linewidth=1.8,
            label=label,
        )
        profile_axis.plot(
            coordinate * 100.0,
            reference_temperature[index],
            "--",
            color=color,
            linewidth=1.1,
        )
    profile_axis.plot([], [], "k-", label="measured")
    profile_axis.plot([], [], "k--", label="exact nonlinear adiabat")
    profile_axis.set_xlabel("x (cm)")
    profile_axis.set_ylabel(r"$T_e$ (eV)")
    profile_axis.set_title("Nonlinear adiabatic compression")
    profile_axis.grid(alpha=0.3)
    profile_axis.legend(fontsize=8, ncol=2)

    sampled_density = density_ratio[significant]
    sampled_temperature = temperature[significant] / args.initial_temperature_eV
    sampled_time = np.broadcast_to(times[:, None], density.shape)[significant]
    scatter = invariant_axis.scatter(
        sampled_density,
        sampled_temperature,
        c=sampled_time * 1.0e6,
        s=8,
        cmap="plasma",
        alpha=0.55,
        label="WarpX",
    )
    density_curve = np.linspace(sampled_density.min(), sampled_density.max(), 300)
    invariant_axis.plot(
        density_curve,
        nonlinear_temperature(density_curve, args) / args.initial_temperature_eV,
        "k-",
        linewidth=2,
        label="exact nonlinear",
    )
    invariant_axis.plot(
        density_curve,
        density_curve ** (args.gamma - 1.0),
        "k--",
        linewidth=1.5,
        label="ideal gas",
    )
    fig.colorbar(scatter, ax=invariant_axis, label=r"time ($\mu$s)")
    invariant_axis.set_xlabel(r"$n/n_0$")
    invariant_axis.set_ylabel(r"$T_e/T_{e0}$")
    invariant_axis.set_title(f"median invariant error {median_invariant_error:.2e}")
    invariant_axis.grid(alpha=0.3)
    invariant_axis.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

    passed = (
        peak_density_swing >= args.min_density_swing
        and median_invariant_error <= args.tol_median_invariant
        and max_invariant_error <= args.tol_max_invariant
        and max_ideal_separation >= args.min_ideal_separation
        and error_ratio_to_ideal <= args.max_error_ratio_to_ideal
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
