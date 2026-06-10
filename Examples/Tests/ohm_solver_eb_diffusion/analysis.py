#!/usr/bin/env python3
#
# --- Analysis script for the hybrid-PIC embedded-boundary diffusion
# --- convergence test. Compares the simulated By field and the plasma
# --- current density (J + J_displacement diagnostics = Ampere current) with
# --- the analytic decaying eigenmode
# ---     By = B1*cos(k*(zr - a/2))*exp(-gamma*t),     k = pi/a,
# ---     J  = curl(B)/mu0 = (B1*k/mu0)*sin(k*(zr - a/2))*exp(-gamma*t) xr_hat,
# --- in the interior of the rotated cavity (a fixed physical margin away from
# --- the wall, so that the same region is sampled at every resolution).
# --- In this configuration the Faraday-push electric field is E = eta*J
# --- identically (holmstrom vacuum handling with rho = 0), so the measured
# --- order of accuracy of J is also the order of accuracy of E.
# --- The conformal (enlarged cell technique) embedded-boundary treatment must
# --- converge at ~2nd order in both B and J, while the stair-step treatment
# --- is limited to ~1st order (with much smaller error constants since the
# --- PEC field boundary condition is enforced at the embedded surface).

import argparse

import numpy as np
import yt
from scipy.constants import mu_0, pi
from scipy.interpolate import RegularGridInterpolator

THETA = np.pi / 8
CAVITY_SIDE = 1.06  # m
HALF_WIDTH = CAVITY_SIDE / 2.0
B1 = 0.01  # T
ETA = 1.0e-3  # Ohm m
DECAY_RATE = ETA / mu_0 * (pi / CAVITY_SIDE) ** 2  # 1/s
KMODE = pi / CAVITY_SIDE
MARGIN = 0.15  # m, interior region samples rotated-frame |x'|,|z'| < a/2 - MARGIN

# rotated-frame unit vectors in the lab (x, z) plane: xhat_r is the outward
# normal of the x' = +a/2 wall, zhat_r is the in-plane tangential direction
XHAT_R = np.array([np.cos(THETA), -np.sin(THETA)])
ZHAT_R = np.array([np.sin(THETA), np.cos(THETA)])


def compute_errors(plotfile):
    """Relative interior L2 errors of By and of the plasma current against the
    decaying eigenmode.

    Returns (err_B, err_J) and the in-plane cell size.
    """
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    t = float(ds.current_time)

    lo = ds.domain_left_edge.to_ndarray()
    hi = ds.domain_right_edge.to_ndarray()
    n = np.array(ds.domain_dimensions)
    dcell = (hi - lo) / n

    x = lo[0] + (np.arange(n[0]) + 0.5) * dcell[0]
    z = lo[2] + (np.arange(n[2]) + 0.5) * dcell[2]
    X, Z = np.meshgrid(x, z, indexing="ij")
    xr = X * np.cos(-THETA) + Z * np.sin(-THETA)
    zr = -X * np.sin(-THETA) + Z * np.cos(-THETA)
    interior = np.maximum(np.abs(xr), np.abs(zr)) < (CAVITY_SIDE / 2 - MARGIN)

    decay = np.exp(-DECAY_RATE * t)
    By_th = B1 * np.cos(KMODE * (zr - CAVITY_SIDE / 2)) * decay
    # analytic plasma current: along the rotated x' direction
    Jxr_th = (B1 * KMODE / mu_0) * np.sin(KMODE * (zr - CAVITY_SIDE / 2)) * decay
    jx_th = Jxr_th * np.cos(THETA)
    jz_th = -Jxr_th * np.sin(THETA)

    By_sim = np.mean(data["By"].to_ndarray(), axis=1)
    jx_sim = np.mean(
        data["jx"].to_ndarray() + data["jx_displacement"].to_ndarray(), axis=1
    )
    jy_sim = np.mean(
        data["jy"].to_ndarray() + data["jy_displacement"].to_ndarray(), axis=1
    )
    jz_sim = np.mean(
        data["jz"].to_ndarray() + data["jz_displacement"].to_ndarray(), axis=1
    )

    err_B = np.sqrt(
        np.sum((By_sim - By_th)[interior] ** 2) / np.sum(By_th[interior] ** 2)
    )
    num_J = np.sum(
        (jx_sim - jx_th)[interior] ** 2
        + jy_sim[interior] ** 2
        + (jz_sim - jz_th)[interior] ** 2
    )
    den_J = np.sum(jx_th[interior] ** 2 + jz_th[interior] ** 2)
    err_J = np.sqrt(num_J / den_J)

    return err_B, err_J, dcell[0]


def wall_bc_residuals(plotfile):
    """PEC boundary-condition residuals of the plasma current at the
    x' = +a/2 wall: the tangential J interpolated to the embedded surface
    (which the boundary condition drives to zero) and the normal gradient of
    the normal J (which must vanish, Neumann). Since the Faraday-push E equals
    eta*J identically in this configuration, these are also the boundary
    condition residuals of E."""
    ds = yt.load(plotfile)
    data = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    lo = ds.domain_left_edge.to_ndarray()
    hi = ds.domain_right_edge.to_ndarray()
    n = np.array(ds.domain_dimensions)
    dcell = (hi - lo) / n
    h = dcell[0]

    x = lo[0] + (np.arange(n[0]) + 0.5) * dcell[0]
    z = lo[2] + (np.arange(n[2]) + 0.5) * dcell[2]
    j = {
        c: np.mean(
            data[f"j{c}"].to_ndarray() + data[f"j{c}_displacement"].to_ndarray(),
            axis=1,
        )
        for c in "xyz"
    }
    interp = {
        c: RegularGridInterpolator((x, z), j[c], bounds_error=True) for c in "xyz"
    }
    scale = max(np.max(np.abs(j[c])) for c in "xyz")

    def sample(zr_val, s):
        xr_val = HALF_WIDTH + s
        pos_x = xr_val * XHAT_R[0] + zr_val * ZHAT_R[0]
        pos_z = xr_val * XHAT_R[1] + zr_val * ZHAT_R[1]
        fx = interp["x"]((pos_x, pos_z))
        fy = interp["y"]((pos_x, pos_z))
        fz = interp["z"]((pos_x, pos_z))
        f_n = fx * XHAT_R[0] + fz * XHAT_R[1]
        f_t1 = fx * ZHAT_R[0] + fz * ZHAT_R[1]
        return np.array([f_n, f_t1, fy])

    jt_wall = []
    djn_dn = []
    for zr_val in np.linspace(-0.3, 0.3, 25):
        f_minus = sample(zr_val, -0.5 * h)
        f_plus = sample(zr_val, 0.5 * h)
        f_wall = 0.5 * (f_minus + f_plus)
        jt_wall.append(np.sqrt(f_wall[1] ** 2 + f_wall[2] ** 2))
        djn_dn.append(abs(f_plus[0] - f_minus[0]) / h)

    return np.max(jt_wall) / scale, np.max(djn_dn) * CAVITY_SIDE / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stair-lo", required=True, help="stair-step, coarse")
    parser.add_argument("--stair-hi", required=True, help="stair-step, fine")
    parser.add_argument("--conformal-lo", required=True, help="conformal, coarse")
    parser.add_argument("--conformal-hi", required=True, help="conformal, fine")
    args = parser.parse_args()

    errB_stair_lo, errJ_stair_lo, h_lo = compute_errors(args.stair_lo)
    errB_stair_hi, errJ_stair_hi, h_hi = compute_errors(args.stair_hi)
    errB_conf_lo, errJ_conf_lo, _ = compute_errors(args.conformal_lo)
    errB_conf_hi, errJ_conf_hi, _ = compute_errors(args.conformal_hi)

    lratio = np.log(h_lo / h_hi)
    orderB_stair = np.log(errB_stair_lo / errB_stair_hi) / lratio
    orderB_conf = np.log(errB_conf_lo / errB_conf_hi) / lratio
    orderJ_stair = np.log(errJ_stair_lo / errJ_stair_hi) / lratio
    orderJ_conf = np.log(errJ_conf_lo / errJ_conf_hi) / lratio

    print(
        f"stair-step: B err {errB_stair_lo:.4e} -> {errB_stair_hi:.4e}, order {orderB_stair:.2f}"
    )
    print(
        f"            J err {errJ_stair_lo:.4e} -> {errJ_stair_hi:.4e}, order {orderJ_stair:.2f}"
    )
    print(
        f"conformal:  B err {errB_conf_lo:.4e} -> {errB_conf_hi:.4e}, order {orderB_conf:.2f}"
    )
    print(
        f"            J err {errJ_conf_lo:.4e} -> {errJ_conf_hi:.4e}, order {orderJ_conf:.2f}"
    )

    # The conformal (enlarged cell technique) update must converge at second
    # order in both B and J (and hence E = eta*J), while the stair-step
    # treatment is limited to first order. Measured values (June 2026, with
    # the PEC field boundary condition active): errB_stair = 2.70e-3/1.37e-3
    # (order 0.98), errJ_stair = 5.87e-3/2.41e-3 (order 1.28),
    # errB_conf = 9.01e-4/1.30e-4 (order 2.79), errJ_conf = 3.99e-3/8.51e-4
    # (order 2.23). Thresholds hold a >= 2x margin against these.
    assert orderB_conf > 1.7, f"conformal B order {orderB_conf:.2f} <= 1.7"
    assert orderJ_conf > 1.8, f"conformal J order {orderJ_conf:.2f} <= 1.8"
    assert orderB_stair < 1.6, f"stair-step B order {orderB_stair:.2f} >= 1.6"
    assert orderJ_stair < 1.7, f"stair-step J order {orderJ_stair:.2f} >= 1.7"
    assert errB_conf_hi < 0.25 * errB_stair_hi, (
        f"conformal B error {errB_conf_hi:.3e} not well below "
        f"stair-step error {errB_stair_hi:.3e}"
    )
    assert errB_conf_hi < 5.0e-4, (
        f"conformal fine-grid B error {errB_conf_hi:.3e} too large"
    )
    assert errJ_conf_hi < 2.5e-3, (
        f"conformal fine-grid J error {errJ_conf_hi:.3e} too large"
    )
    assert errB_stair_lo < 1.0e-2, (
        f"stair-step coarse-grid B error {errB_stair_lo:.3e} too large"
    )
    assert errJ_stair_lo < 2.0e-2, (
        f"stair-step coarse-grid J error {errJ_stair_lo:.3e} too large"
    )

    # PEC boundary-condition residuals at the embedded surface: tangential J
    # (and hence tangential E = eta*J) must vanish at the wall and the normal
    # component must have zero normal gradient, both converging at least at
    # 2nd order with resolution. Measured (June 2026): stair Jt 7.37e-2 ->
    # 1.77e-3 (ratio 42), dJn 7.6 -> 5.0e-2; conformal Jt 7.82e-2 -> 4.52e-3
    # (ratio 17), dJn 7.6 -> 1.2e-1. A 2nd-order boundary condition gives a
    # ratio of 4 between these resolutions; thresholds keep >= 2x margins.
    for label, lo_path, hi_path, jt_hi_cap in (
        ("stair-step", args.stair_lo, args.stair_hi, 8.0e-3),
        ("conformal", args.conformal_lo, args.conformal_hi, 2.0e-2),
    ):
        jt_lo, djn_lo = wall_bc_residuals(lo_path)
        jt_hi, djn_hi = wall_bc_residuals(hi_path)
        print(
            f"{label}: wall |J_t|/scale {jt_lo:.3e} -> {jt_hi:.3e} "
            f"(ratio {jt_lo / jt_hi:.1f}), |dJ_n/dn|*a/scale {djn_lo:.3e} -> {djn_hi:.3e}"
        )
        assert jt_lo / jt_hi > 3.0, (
            f"{label}: tangential J at the wall not converging at ~2nd order"
        )
        assert djn_lo / djn_hi > 3.0, (
            f"{label}: normal-J gradient at the wall not converging at ~2nd order"
        )
        assert jt_hi < jt_hi_cap, f"{label}: tangential J at the wall too large"


if __name__ == "__main__":
    main()
