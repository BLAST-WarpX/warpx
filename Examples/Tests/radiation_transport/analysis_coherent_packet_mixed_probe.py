#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
# License: BSD-3-Clause-LBNL
"""Actual native packet/ion inventories; no scalar or moment-field oracle.

The stable kinetic-energy difference and carry units match the independent
flux-relaxed roundtrip analysis, but this loader deliberately requires only
packet-mode fields and the native ``streaming`` particle carry path.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yt

LD = np.longdouble
C = LD("299792458")
MP = LD("1.67262192595e-27")
KB = LD("1.380649e-23")
QE = LD("1.602176634e-19")
EPS0 = LD("8.8541878188e-12")
DT = LD("1e-11")
DIRECTION = np.array([LD("0.3"), LD("0.4"), np.sqrt(LD("0.75"))])
yt.funcs.mylog.setLevel(40)


def plotfiles(directory):
    return sorted(
        p for p in (directory / "diags").glob("diag1*") if p.name[5:].isdigit()
    )


def load(path):
    ds = yt.load(str(path))
    data = ds.all_data()
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)

    def species(name):
        ids = np.asarray(data[name, "particle_id"].v, dtype=np.int64)
        cpu = np.asarray(data[name, "particle_cpu"].v, dtype=np.int64)
        order = np.lexsort((cpu, ids))

        def component(key):
            return np.asarray(data[name, "particle_" + key].v, dtype=LD)[order]

        result = dict(
            ids=np.stack((ids[order], cpu[order]), axis=1),
            weight=component("weight"),
            momentum=np.stack([component("momentum_" + a) for a in "xyz"], axis=1),
            position=np.stack([component("position_" + a) for a in "xyz"], axis=1),
        )
        assert len({tuple(pair) for pair in result["ids"]}) == len(ids)
        for key in ("weight", "momentum", "position"):
            assert np.all(np.isfinite(result[key])), (path, name, key)
        assert np.all(result["weight"] > 0)
        if name == "ions":
            result["carry"] = np.stack(
                [component("radiation_impulse_streaming_u" + a) for a in "xyz"], axis=1
            )
            result["work_carry"] = component("radiation_impulse_streaming_work")
            assert np.all(np.isfinite(result["carry"]))
            assert np.all(np.isfinite(result["work_carry"]))
        return result

    def field(name):
        values = np.asarray(grid["boxlib", name].v, dtype=LD).ravel()
        assert np.all(np.isfinite(values)), (path, name)
        return values

    # These states do not exist in physical packet-only mode.
    assert ("boxlib", "radiation_diffusion_energy") not in ds.field_list
    assert not any("radiation_flux_relaxed_q" in key[1] for key in ds.field_list)
    photons, ions = species("photons"), species("ions")
    photon_e = C * np.sum(
        photons["weight"] * np.sqrt(np.sum(photons["momentum"] ** 2, axis=1))
    )
    photon_p = np.sum(photons["weight"][:, None] * photons["momentum"], axis=0)
    rho, temperature = field("rho"), field("Te")
    volume = np.prod(np.asarray(ds.domain_width.v, dtype=LD))
    internal_proxy = np.mean(rho / QE * LD("1.5") * KB * temperature) * volume
    # Ideal gas, gamma=5/3: pressure is linear in internal energy. Diagnostics
    # average nodal fields separately; multiplying averaged rho and Te loses
    # their covariance. Averaging Pe preserves the native caloric identity.
    internal = LD("1.5") * np.mean(field("Pe")) * volume
    electric = np.stack([field("E" + a) for a in "xyz"], axis=1)
    magnetic = np.stack([field("B" + a) for a in "xyz"], axis=1)
    electric_energy = EPS0 / 2 * np.mean(np.sum(electric**2, axis=1)) * volume
    magnetic_energy = EPS0 * C**2 / 2 * np.mean(np.sum(magnetic**2, axis=1)) * volume
    field_momentum = EPS0 * np.mean(np.cross(electric, magnetic), axis=0) * volume
    return dict(
        step=int(path.name[5:]),
        time=LD(ds.current_time.v),
        photons=photons,
        ions=ions,
        energy=photon_e,
        momentum=photon_p,
        rho=rho,
        temperature=temperature,
        internal=internal,
        internal_proxy=internal_proxy,
        electric_energy=electric_energy,
        magnetic_energy=magnetic_energy,
        field_momentum=field_momentum,
        material_heat=field("radiation_material_energy"),
    )


def manifest(directory, step):
    checkpoint = directory / "diags" / f"chk{step:06d}"
    words = (checkpoint / "RadiationCoherentPackets_data.txt").read_text().split()
    assert words == ["chronological_coherent_packets_v1", str(step)], words
    # The fixed problem needs one predetermined chronological segment per step.
    # Also require the independent resolved-seed and particle-carry manifests.
    accounting = (checkpoint / "RadiationTransport_data.txt").read_text().split()
    assert len(accounting) == 3 and accounting[2] == "42", accounting
    carry = (checkpoint / "RadiationParticleCarry_data.txt").read_text().split()
    assert carry == ["particle_carry_v1", "1", "1", "ions"], carry
    return int(words[1])


def survival_bounds(tau, beta_bound):
    """Poisson-binomial count envelope, including bounded proper-rate motion.

    Every uncollided trajectory is a known straight ray. For |beta|<=B, its
    proper-rate factor lies between sqrt((1-B)/(1+B)) and its reciprocal.
    Six standard deviations plus one count are fixed before executing a run.
    """
    slow = np.sqrt((1 - beta_bound) / (1 + beta_bound))
    fast = 1 / slow
    lower = np.exp(-fast * tau)
    upper = np.exp(-slow * tau)
    worst_variance = np.maximum(lower * (1 - lower), upper * (1 - upper))
    worst_variance[(lower <= LD("0.5")) & (upper >= LD("0.5"))] = LD("0.25")
    margin = 6 * np.sqrt(np.sum(worst_variance)) + 1
    return max(LD(0), np.sum(lower) - margin), min(LD(len(tau)), np.sum(upper) + margin)


def optical_depth(initial_z, time):
    """Exact straight-ray column through the periodic slab [0.5, 0.75)."""

    def column(z):
        periods = np.floor(z)
        return periods / 4 + np.clip(z - periods - LD("0.5"), LD(0), LD("0.25"))

    final_z = initial_z + C * DIRECTION[2] * time
    return 200 * (column(final_z) - column(initial_z)) / DIRECTION[2]


def compare_restart(row, reference):
    for name in ("ions", "photons"):
        current, prior = row[name], reference[name]
        assert np.array_equal(current["ids"], prior["ids"])
        assert np.array_equal(current["weight"], prior["weight"])
        assert np.max(np.abs(current["position"] - prior["position"])) < LD("1e-11")
        if name == "ions":
            assert np.max(np.abs(current["momentum"] - prior["momentum"])) / MP < LD(
                "1e-8"
            )
            assert np.max(np.abs(current["carry"] - prior["carry"])) < LD("1e-8")
            assert np.max(np.abs(current["work_carry"] - prior["work_carry"])) < LD(
                "1e-8"
            )
        else:
            scale = np.sqrt(np.sum(prior["momentum"] ** 2, axis=1))
            error = np.sqrt(
                np.sum((current["momentum"] - prior["momentum"]) ** 2, axis=1)
            )
            assert np.max(error / scale) < LD("1e-11")
    assert abs(row["energy"] - reference["energy"]) < LD("1e-11") * reference["energy"]


def check(
    directory,
    reference_directory=None,
    *,
    steps=80,
    density=LD("1e20"),
    initial_u=(0, 0, 0),
    work_sign="+",
):
    if steps < 80 or steps % 10:
        raise ValueError("steps must be a multiple of 10, at least 80")
    density = LD(density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")
    initial_u = np.asarray(initial_u, dtype=LD)
    if initial_u.shape != (3,) or not np.all(np.isfinite(initial_u)):
        raise ValueError(
            "initial-u requires three finite proper-velocity components / c"
        )
    if work_sign not in ("+", "-"):
        raise ValueError("work-sign must be + or -")
    expected_steps = list(range(0, steps + 1, 10))
    paths = plotfiles(directory)
    if not paths:
        raise RuntimeError("No packet-mode plotfiles; an incomplete run cannot pass")
    current_rows = [load(path) for path in paths]
    rows = current_rows
    if reference_directory is not None:
        prior = [load(path) for path in plotfiles(reference_directory)]
        assert [row["step"] for row in prior] == expected_steps
        assert current_rows[0]["step"] in (40, 50)
        by_step = {row["step"]: row for row in prior}
        for row in current_rows:
            compare_restart(row, by_step[row["step"]])
        rows = [
            row for row in prior if row["step"] < current_rows[0]["step"]
        ] + current_rows
        manifest(reference_directory, 40)
    assert [row["step"] for row in rows] == expected_steps, [
        row["step"] for row in rows
    ]
    initial = rows[0]
    packets0, ions0 = initial["photons"], initial["ions"]
    assert len(packets0["ids"]) == 512
    assert len(ions0["ids"]) == 4096
    assert abs(initial["energy"] / 1250 - 1) < LD("1e-12")
    pnorm0 = np.sqrt(np.sum(packets0["momentum"] ** 2, axis=1))
    assert np.max(np.abs(packets0["momentum"] / pnorm0[:, None] - DIRECTION)) < LD(
        "1e-12"
    )
    assert np.all(
        (packets0["position"][:, 2] >= LD("0.375"))
        & (packets0["position"][:, 2] < LD("0.5"))
    )
    mass = MP * ions0["weight"]
    assert abs(np.sum(mass) / (MP * density) - 1) < LD("1e-12")
    u0 = ions0["momentum"] / MP
    for axis in range(3):
        expected_u = initial_u[axis] * C
        if expected_u == 0:
            assert np.all(u0[:, axis] == 0)
        else:
            # Input parsing, multiplication by c and diagnostic mass conversion
            # each round to native double precision; this is not a fitted bound.
            assert np.max(np.abs(u0[:, axis] - expected_u)) <= (
                16 * np.finfo(np.float64).eps * abs(expected_u)
            )
    gamma0 = np.sqrt(1 + np.sum(u0**2, axis=1) / C**2)
    initial_kinetic = np.sum(mass * np.sum(u0**2, axis=1) / (gamma0 + 1))
    # A conservative energy-budget speed envelope, not a fitted source rate.
    # The extra 1% also bounds the explicitly checked signed carry inventory.
    budget = LD("1.01") * (
        initial["energy"] + abs(initial["internal"]) + initial_kinetic
    )
    gamma_bound = 1 + budget / (np.min(mass) * C**2)
    beta_bound = np.sqrt(1 - 1 / gamma_bound**2)
    history = []
    for row in rows:
        packets, ions = row["photons"], row["ions"]
        for current, original in ((packets, packets0), (ions, ions0)):
            assert np.array_equal(current["ids"], original["ids"])
            assert np.array_equal(current["weight"], original["weight"])
        assert abs(row["time"] - row["step"] * DT) < LD("1e-20")
        u = ions["momentum"] / MP
        du = u - u0
        gamma = np.sqrt(1 + np.sum(u**2, axis=1) / C**2)
        kinetic = np.sum(mass * np.sum(du * (u + u0), axis=1) / (gamma + gamma0))
        carry_work = np.sum(mass * (ions["work_carry"] - ions0["work_carry"]))
        assert (
            np.sum(np.abs(mass * ions["work_carry"])) < LD("0.01") * initial["energy"]
        )
        actual_p = np.sum(mass[:, None] * du, axis=0)
        carry_p = np.sum(mass[:, None] * (ions["carry"] - ions0["carry"]), axis=0)
        thermal_change = row["internal"] - initial["internal"]
        electric_change = row["electric_energy"] - initial["electric_energy"]
        magnetic_change = row["magnetic_energy"] - initial["magnetic_energy"]
        # Hybrid's closed discrete field reservoir is magnetic. Also report
        # and bound the independently measured electric diagnostic, so a hidden
        # field-energy change cannot masquerade as a radiation-only balance.
        energy_error = (
            row["energy"]
            - initial["energy"]
            + kinetic
            + carry_work
            + thermal_change
            + magnetic_change
        )
        momentum_error = (
            row["momentum"]
            - initial["momentum"]
            + actual_p
            + carry_p
            + row["field_momentum"]
            - initial["field_momentum"]
        )
        assert abs(energy_error) < LD("1e-10") * initial["energy"]
        assert abs(energy_error + electric_change) < LD("1e-10") * initial["energy"]
        assert np.max(np.abs(momentum_error)) < LD("1e-10") * initial["energy"] / C
        assert np.all(row["material_heat"] == 0)
        speed = np.sqrt(np.sum(u**2, axis=1)) / (gamma * C)
        assert np.max(speed) <= beta_bound
        uncollided = np.all(packets["momentum"] == packets0["momentum"], axis=1)
        tau = optical_depth(packets0["position"][:, 2], row["time"])
        lower, upper = survival_bounds(tau, beta_bound)
        count = int(np.count_nonzero(uncollided))
        assert lower <= count <= upper, (row["step"], count, lower, upper)
        ballistic = (packets0["position"] + C * DIRECTION * row["time"]) % 1
        if count:
            delta = np.abs(packets["position"][uncollided] - ballistic[uncollided])
            assert np.max(np.minimum(delta, 1 - delta)) < LD("1e-12")
        if row["step"] <= 10:
            # Normal hybrid ion pushing is required by the native pressure-work
            # contract. Its equilibrium roundoff is covered by the full energy
            # and momentum balances, not mistaken for a radiation interaction.
            assert count == 512
            assert np.array_equal(ions["carry"], ions0["carry"])
            assert np.array_equal(ions["work_carry"], ions0["work_carry"])
        history.append(
            dict(
                step=row["step"],
                photons=len(packets["ids"]),
                photon_energy_J=float(row["energy"]),
                ion_kinetic_gain_J=float(kinetic),
                work_carry_change_J=float(carry_work),
                ion_momentum_gain_kgms=list(map(float, actual_p)),
                momentum_carry_change_kgms=list(map(float, carry_p)),
                raw_energy_error_J=float(energy_error),
                raw_momentum_error_kgms=list(map(float, momentum_error)),
                electron_thermal_change_J=float(thermal_change),
                electron_thermal_proxy_change_J=float(
                    row["internal_proxy"] - initial["internal_proxy"]
                ),
                electron_thermal_proxy_offset_change_J=float(
                    row["internal_proxy"] - initial["internal_proxy"] - thermal_change
                ),
                electric_energy_change_J=float(electric_change),
                magnetic_energy_change_J=float(magnetic_change),
                uncollided=count,
                stationary_expected=float(np.sum(np.exp(-tau))),
                proper_rate_count_bounds=[float(lower), float(upper)],
                maximum_ion_beta=float(np.max(speed)),
            )
        )
    sign = 1 if work_sign == "+" else -1
    assert sign * history[-1]["ion_kinetic_gain_J"] > 0
    if work_sign == "-":
        assert rows[-1]["energy"] > initial["energy"]
    assert np.all(np.asarray(history[-1]["ion_momentum_gain_kgms"]) > 0)
    assert history[-1]["uncollided"] <= 2
    manifest(directory, steps)
    if reference_directory is None:
        manifest(directory, 40)
    result = dict(
        model="chronological physical coherent packets; no diffusion/conversion",
        beta_rate_envelope=float(beta_bound),
        restart=reference_directory is not None,
        steps=steps,
        density_m3=float(density),
        initial_u_over_c=list(map(float, initial_u)),
        expected_work_sign=work_sign,
        initial_ion_kinetic_J=float(initial_kinetic),
        history=history,
    )
    (directory / "coherent_packet_mixed_probe_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--density", type=LD, default=LD("1e20"))
    parser.add_argument("--initial-u", nargs=3, type=LD, default=(0, 0, 0))
    parser.add_argument("--work-sign", choices=("+", "-"), default="+")
    args = parser.parse_args()
    check(
        args.directory,
        args.reference,
        steps=args.steps,
        density=args.density,
        initial_u=args.initial_u,
        work_sign=args.work_sign,
    )
