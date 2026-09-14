#!/usr/bin/env python3
"""Validate nonlinear-caloric Qei exchange and its conjugate ion ledger.

The independent recurrence freezes both heat capacities over each source
step, applies the exact two-temperature exchange used by WarpX, then inverts
the manufactured latent electron caloric EOS.  The test compares both
temperatures and verifies conservation of electron caloric plus ion thermal
energy; it does not use an ideal-gas exponential surrogate.
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
K_PER_EV = Q_E / K_B


def domain_means(diag_dir):
    series = OpenPMDTimeSeries(str(diag_dir))
    times = np.asarray(series.t, dtype=float)
    electron_temperature = []
    ion_temperature = []
    for iteration in series.iterations:
        te, _ = series.get_field("Te", iteration=iteration)
        ti, _ = series.get_field("T_ions", iteration=iteration)
        rho, _ = series.get_field("rho", iteration=iteration)
        weight = np.asarray(rho, dtype=float) / Q_E
        weight_sum = float(np.sum(weight))
        electron_temperature.append(
            float(np.sum(np.asarray(te, dtype=float) * weight) / weight_sum) / K_PER_EV
        )
        ion_temperature.append(
            float(np.sum(np.asarray(ti, dtype=float) * weight) / weight_sum)
        )
    return (
        times,
        np.asarray(electron_temperature),
        np.asarray(ion_temperature),
    )


def latent_fraction(temperature, transition_temperature, sharpness):
    ratio = np.asarray(temperature, dtype=float) / transition_temperature
    return ratio**sharpness / (1.0 + ratio**sharpness)


def electron_energy(temperature, gamma, latent_energy, transition, sharpness):
    return np.asarray(temperature, dtype=float) / (gamma - 1.0) + (
        latent_energy * latent_fraction(temperature, transition, sharpness)
    )


def electron_capacity(temperature, gamma, latent_energy, transition, sharpness):
    fraction = latent_fraction(temperature, transition, sharpness)
    return 1.0 / (gamma - 1.0) + (
        latent_energy * sharpness * fraction * (1.0 - fraction) / temperature
    )


def invert_electron_energy(energy, gamma, latent_energy, transition, sharpness):
    lower = 0.0
    upper = max((gamma - 1.0) * energy, transition)
    for _ in range(120):
        middle = 0.5 * (lower + upper)
        if (
            electron_energy(middle, gamma, latent_energy, transition, sharpness)
            < energy
        ):
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def advance_reference(te, ti, dt, nu, gamma, latent_energy, transition, sharpness):
    ce = float(electron_capacity(te, gamma, latent_energy, transition, sharpness))
    ci = 1.5
    conductance = 3.0 * nu
    coupled_rate = conductance * (1.0 / ce + 1.0 / ci)
    fraction = -np.expm1(-coupled_rate * dt)
    series_capacity = ce * ci / (ce + ci)
    electron_delta = series_capacity * (ti - te) * fraction
    target_energy = (
        float(electron_energy(te, gamma, latent_energy, transition, sharpness))
        + electron_delta
    )
    return (
        invert_electron_energy(
            target_energy, gamma, latent_energy, transition, sharpness
        ),
        ti - electron_delta / ci,
    )


def reference_at_diagnostics(times, te0, ti0, dt, args):
    te, ti = te0, ti0
    model_te = [te]
    model_ti = [ti]
    previous_step = 0
    for time in times[1:]:
        target_step = int(round(time / dt))
        for _ in range(previous_step, target_step):
            te, ti = advance_reference(
                te,
                ti,
                dt,
                args.nu_ei,
                args.gamma,
                args.latent_energy_eV,
                args.transition_temperature_eV,
                args.latent_sharpness,
            )
        previous_step = target_step
        model_te.append(te)
        model_ti.append(ti)
    return np.asarray(model_te), np.asarray(model_ti)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", default="diags/field_diags")
    parser.add_argument("--nu-ei", type=float, default=1.0e6)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--latent-energy-eV", type=float, default=1000.0)
    parser.add_argument("--transition-temperature-eV", type=float, default=200.0)
    parser.add_argument("--latent-sharpness", type=float, default=4.0)
    parser.add_argument("--steps-per-electron-tau", type=int, default=100)
    parser.add_argument("--temperature-rtol", type=float, default=0.05)
    parser.add_argument("--energy-rtol", type=float, default=0.02)
    parser.add_argument("--out", default="latent_qei_check.png")
    args = parser.parse_args(argv)

    times, te, ti = domain_means(args.diag_dir)
    electron_sink_rate = 3.0 * (args.gamma - 1.0) * args.nu_ei
    dt = 1.0 / electron_sink_rate / args.steps_per_electron_tau
    model_te, model_ti = reference_at_diagnostics(times, te[0], ti[0], dt, args)

    temperature_scale = max(abs(te[0] - ti[0]), 1.0)
    te_error = float(np.max(np.abs(te - model_te)) / temperature_scale)
    ti_error = float(np.max(np.abs(ti - model_ti)) / temperature_scale)
    total_energy = (
        electron_energy(
            te,
            args.gamma,
            args.latent_energy_eV,
            args.transition_temperature_eV,
            args.latent_sharpness,
        )
        + 1.5 * ti
    )
    energy_drift = (total_energy - total_energy[0]) / total_energy[0]
    maximum_energy_drift = float(np.max(np.abs(energy_drift)))

    print("Nonlinear fixed-charge Qei caloric-source regression")
    print(f"  max Te/reference error: {te_error:.6e}")
    print(f"  max Ti/reference error: {ti_error:.6e}")
    print(f"  max total-energy drift: {maximum_energy_drift:.6e}")

    time_microseconds = times * 1.0e6
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(time_microseconds, te, "o", label="WarpX Te")
    axes[0].plot(time_microseconds, model_te, "-", label="analytic-step Te")
    axes[0].plot(time_microseconds, ti, "s", label="WarpX Ti")
    axes[0].plot(time_microseconds, model_ti, "--", label="analytic-step Ti")
    axes[0].set_xlabel("time [microseconds]")
    axes[0].set_ylabel("temperature [eV]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(time_microseconds, energy_drift, "o-")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time [microseconds]")
    axes[1].set_ylabel("relative total caloric-energy drift")
    axes[1].grid(alpha=0.3)
    figure.suptitle("Latent-electron EOS: conservative Qei exchange")
    figure.tight_layout()
    figure.savefig(args.out, dpi=150)

    passed = (
        te_error <= args.temperature_rtol
        and ti_error <= args.temperature_rtol
        and maximum_energy_drift <= args.energy_rtol
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
