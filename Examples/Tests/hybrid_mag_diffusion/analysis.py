#!/usr/bin/env python3
"""
Analytic check for constant-eta magnetic diffusion of a Fourier By mode.

Each PIC step applies two magnetic-diffusion half-steps with amplification

    g = (1 - (1-theta) * lambda * dt_half) / (1 + theta * lambda * dt_half)

where lambda is the second-order finite-difference Fourier symbol. The
continuous exp(-chi * kx^2 * t) decay is also reported as a diagnostic.

Reads By min/max from the AMReX plotfile Cell_H (no yt/openPMD).
Passes if the relative amplitude error versus continuous exponential decay
of the measured t=0 peak is below 1%.
"""

import glob
import os
import re
import sys

import numpy as np

MU0 = 1.2566370612685e-06


def get_input_value(inputs, key):
    match = re.search(
        rf"^{re.escape(key)}\s*=\s*(.*?)\s*(?:#.*)?$", inputs, re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"Could not find {key} in warpx_used_inputs")
    return match.group(1).split()


def get_runtime_parameters():
    with open("warpx_used_inputs") as infile:
        inputs = infile.read()

    prob_lo = get_input_value(inputs, "geometry.prob_lo")
    prob_hi = get_input_value(inputs, "geometry.prob_hi")
    return {
        "dt": float(get_input_value(inputs, "warpx.const_dt")[0]),
        "eta": float(
            get_input_value(inputs, "hybrid_pic_model.mag_diff_constant_eta")[0]
        ),
        "nsteps": int(get_input_value(inputs, "max_step")[0]),
        "n_cell_x": int(get_input_value(inputs, "amr.n_cell")[0]),
        "theta": float(get_input_value(inputs, "hybrid_pic_model.mag_diff_theta")[0]),
        "lx": float(prob_hi[0]) - float(prob_lo[0]),
    }


def by_minmax_from_diag(diag_dir):
    """Return the By extrema using the component order in the plotfile header."""
    header = os.path.join(diag_dir, "Header")
    with open(header) as fh:
        header_lines = [line.strip() for line in fh]
    ncomp = int(header_lines[1])
    component_names = header_lines[2 : 2 + ncomp]
    try:
        by_component = component_names.index("By")
    except ValueError as exc:
        raise RuntimeError(f"By is absent from {header}: {component_names}") from exc

    cell_h = os.path.join(diag_dir, "Level_0", "Cell_H")
    if not os.path.isfile(cell_h):
        raise FileNotFoundError(cell_h)
    with open(cell_h) as fh:
        text = fh.read()
    # VisMF trailing lines: "1,ncomp" then mins, blank, "1,ncomp" then maxs
    # Find last two float-list lines with 3 comma-separated values
    float_lines = []
    for line in text.splitlines():
        if re.match(r"^-?[0-9]", line.strip()):
            parts = [p for p in line.strip().rstrip(",").split(",") if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) >= ncomp:
                float_lines.append(vals)
    if len(float_lines) < 2:
        raise RuntimeError(f"Could not parse By min/max from {cell_h}:\n{text}")
    # mins then maxs
    mins, maxs = float_lines[-2], float_lines[-1]
    return mins[by_component], maxs[by_component]


def main():
    parameters = get_runtime_parameters()
    eta = parameters["eta"]
    theta = parameters["theta"]
    dt = parameters["dt"]
    nsteps = parameters["nsteps"]
    lx = parameters["lx"]
    kx = 2.0 * np.pi / lx
    chi = eta / MU0
    t_end = nsteps * dt
    dx = lx / parameters["n_cell_x"]
    lambda_fd = chi * 4.0 * np.sin(0.5 * kx * dx) ** 2 / dx**2
    z = lambda_fd * 0.5 * dt
    amplification = (1.0 - (1.0 - theta) * z) / (1.0 + theta * z)
    discrete_decay = abs(amplification) ** (2 * nsteps)
    continuous_decay = np.exp(-chi * kx * kx * t_end)

    candidates = sorted(glob.glob("diags/diag1*"))
    if not candidates:
        print("FAILED: no diags/diag1* plotfiles")
        sys.exit(1)

    # First and last dumps
    first = candidates[0]
    last = candidates[-1]
    print(f"first={first} last={last}")

    bmin0, bmax0 = by_minmax_from_diag(first)
    bmin1, bmax1 = by_minmax_from_diag(last)
    amp0 = max(abs(bmin0), abs(bmax0))
    amp1 = max(abs(bmin1), abs(bmax1))

    expected_discrete = amp0 * discrete_decay
    expected_continuous = amp0 * continuous_decay
    rel_discrete_err = abs(amp1 - expected_discrete) / max(amp0, 1e-30)
    rel_continuous_err = abs(amp1 - expected_continuous) / max(amp0, 1e-30)

    print(f"theta = {theta:.6f}  chi = {chi:.6e} m^2/s  kx = {kx:.6e} 1/m")
    print(f"t = {t_end:.6e} s  lambda_fd = {lambda_fd:.6e} 1/s")
    print(f"half-step amplification = {amplification:.8f}")
    print(f"discrete decay factor = {discrete_decay:.8f}")
    print(f"continuous decay factor = {continuous_decay:.8f}")
    print(f"By amp t=0   = {amp0:.8e} T")
    print(f"By amp t_end = {amp1:.8e} T")
    print(f"discrete expected   = {expected_discrete:.8e} T")
    print(f"continuous expected = {expected_continuous:.8e} T")
    print(f"discrete rel error   = {rel_discrete_err:.4e}")
    print(f"continuous rel error = {rel_continuous_err:.4e}")

    if amp0 < 1e-8:
        print("FAILED: initial By amplitude too small (B init missing?)")
        sys.exit(1)

    if amp1 > 0.999 * amp0 and discrete_decay < 0.999:
        print("FAILED: amplitude did not decay")
        sys.exit(1)

    tol = 1.0e-4
    if rel_discrete_err > tol:
        print(f"FAILED: discrete relative error {rel_discrete_err} > {tol}")
        sys.exit(1)

    print("PASSED: hybrid magnetic diffusion theta-method amplitude")
    sys.exit(0)


if __name__ == "__main__":
    main()
