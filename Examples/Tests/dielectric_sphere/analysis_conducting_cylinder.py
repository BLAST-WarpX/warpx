#!/usr/bin/env python3

import sys

import numpy as np
from conducting_cylinder_solution import (
    EPSILON_R,
    ProblemParameters,
    analytic_solution,
    compute_error_metrics,
    load_plotfile,
    make_error_masks,
)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "diags/diag1000001"

    data = load_plotfile(filename)
    params = ProblemParameters()
    exact = analytic_solution(data, params)
    masks = make_error_masks(data, params)
    metrics = compute_error_metrics(data, exact, masks)

    print(f"Field RMS relative error: {metrics.field_rms_relative_error:.6e}")
    print(
        f"Shell field RMS relative error: {metrics.shell_field_rms_relative_error:.6e}"
    )
    print(f"Phi RMS relative error: {metrics.phi_rms_relative_error:.6e}")

    assert metrics.field_rms_relative_error < 0.03
    assert metrics.shell_field_rms_relative_error < 0.06
    assert metrics.phi_rms_relative_error < 0.03
    assert np.max(np.abs(data.ey)) < 1.0e-14

    assert np.allclose(data.phi[masks.deep_core], 0.0)
    assert np.allclose(data.ex[masks.deep_core], 0.0)
    assert np.allclose(data.ez[masks.deep_core], 0.0)
    assert np.allclose(data.eb_covered[masks.deep_core], 1.0)
    assert np.allclose(data.eb_covered[masks.outside_conductor], 0.0)

    assert np.allclose(data.epsilon[masks.deep_shell], EPSILON_R)
    assert np.allclose(data.epsilon[masks.far_outside], 1.0)

    assert np.allclose(data.dielectric_mask[masks.inside_dielectric_object], 1.0)
    assert np.allclose(data.dielectric_mask[masks.far_outside], 0.0)
    assert np.all(
        (data.dielectric_mask >= -1.0e-12) & (data.dielectric_mask <= 1.0 + 1.0e-12)
    )


if __name__ == "__main__":
    main()
