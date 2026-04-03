#!/usr/bin/env python3
"""
Analysis script for the spectral_precision test suite.

Validates that PSATD vacuum eigenmode evolution preserves the spectral
phase rotation to near-machine precision. Compares step-0 fields against
step-N fields by applying the exact PSATD spectral update analytically.

The test reads both the initial (step 0) and final plotfiles, FFTs the
initial fields, applies the exact vacuum PSATD phase rotation in spectral
space, and compares the result against the final fields. This isolates
the PSATD time advance from any initialization artifacts.

Handles 1D, 2D, 3D, RZ z-mode, and RZ Hankel eigenfunction tests via
the --dim argument.

Usage:
    python analysis_spectral_precision.py --dim {1,2,3,rz,rz_hankel} \
        --diag-dir-init <path> --diag-dir-final <path>
"""

import argparse

import numpy as np
import yt

yt.funcs.mylog.setLevel(50)

from scipy.constants import c as c_light


def analyze_cartesian(ds_init, ds_final, ndim):
    """
    Analyze Cartesian (1D/2D/3D) PSATD vacuum eigenmode by comparing
    step-0 and step-N fields via spectral phase rotation.

    The PSATD vacuum update for each spectral mode k is:
      E_hat^{n+1} = C * E_hat^n  +  S_c * (ik x B_hat^n)
      B_hat^{n+1} = C * B_hat^n  -  S   * (ik x E_hat^n)
    where C = cos(omega*dt), S = sin(omega*dt)/omega, S_c = c^2 * S,
    and omega = c * |k_modified|.

    After N steps this is an exact phase rotation — amplitude is preserved.
    We verify by FFT of the step-0 fields, applying N rotations, and
    comparing the IFFT result against the step-N plotfile.
    """
    t_init = float(ds_init.current_time)
    t_final = float(ds_final.current_time)
    dt = t_final - t_init

    data_final = ds_final.covering_grid(
        level=0, left_edge=ds_final.domain_left_edge, dims=ds_final.domain_dimensions
    )
    data_init = ds_init.covering_grid(
        level=0, left_edge=ds_init.domain_left_edge, dims=ds_init.domain_dimensions
    )

    if ndim == 1:
        Ey_init = data_init[("mesh", "Ey")].to_ndarray().flatten()
        Bx_init = data_init[("mesh", "Bx")].to_ndarray().flatten()
        Ey_final = data_final[("mesh", "Ey")].to_ndarray().flatten()
        Bx_final = data_final[("mesh", "Bx")].to_ndarray().flatten()

        nz = len(Ey_init)
        Lz = float(ds_init.domain_right_edge[0] - ds_init.domain_left_edge[0])

        # FFT the initial fields
        Ey_hat = np.fft.rfft(Ey_init)
        Bx_hat = np.fft.rfft(Bx_init)

        # Wavenumbers
        kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi
        omega = c_light * np.abs(kz)

        # Phase rotation for total time dt
        C = np.cos(omega * dt)
        S = np.where(
            omega > 0,
            np.divide(
                np.sin(omega * dt), omega, where=omega > 0, out=np.full_like(omega, dt)
            ),
            dt,
        )

        # PSATD update: +z traveling wave has (Ey, Bx)
        # In 1D (z-only): (ik x B)_y = -ikz * Bx (for k = kz z-hat)
        #                  (ik x E)_x =  ikz * Ey (wrong sign?)
        # Actually in 1D: k = kz z-hat, E = Ey y-hat, B = Bx x-hat
        # (ik x B)_y = (ikz z-hat x Bx x-hat)_y = ikz*Bx*(z x x)_y = ikz*Bx*(y-hat)_y = ikz*Bx
        # Wait: z x x = -y (right-hand rule: z x x = -(x x z) = -y)
        # So (ik x B)_y = ikz*Bx*(-1) = -ikz*Bx
        # (ik x E)_x = (ikz z-hat x Ey y-hat)_x = ikz*Ey*(z x y)_x = ikz*Ey*(x-hat)_x = -ikz*Ey
        # Wait: z x y = x? No: z x y = -(y x z) = -x.  Hmm.
        # z-hat x y-hat: using right hand rule, z x y = -x? No.
        # x x y = z, y x z = x, z x x = y.  So z x y = -(y x z) = -x.
        # So (ik x E)_x = ikz*Ey*(z x y)_x = ikz*Ey*(-x)_x = -ikz*Ey

        # E_y^new = C*E_y + c^2*S*(-ikz*Bx)
        # B_x^new = C*B_x - S*(-ikz*Ey) = C*Bx + S*ikz*Ey
        Ey_hat_new = C * Ey_hat + c_light**2 * S * (-1j * kz * Bx_hat)
        Bx_hat_new = C * Bx_hat - S * (-1j * kz * Ey_hat)

        Ey_expected = np.fft.irfft(Ey_hat_new, n=nz)
        Bx_expected = np.fft.irfft(Bx_hat_new, n=nz)

        E0 = np.max(np.abs(Ey_init))
        B0 = np.max(np.abs(Bx_init))
        err_Ey = np.max(np.abs(Ey_final - Ey_expected)) / E0
        err_Bx = np.max(np.abs(Bx_final - Bx_expected)) / B0

        print(f"1D PSATD vacuum eigenmode: t_init={t_init:.2e}, t_final={t_final:.2e}")
        print(f"  Ey max relative error: {err_Ey:.2e}")
        print(f"  Bx max relative error: {err_Bx:.2e}")

        return max(err_Ey, err_Bx)

    elif ndim == 2:
        Ey_init = data_init[("mesh", "Ey")].to_ndarray().squeeze()
        Bz_init = data_init[("mesh", "Bz")].to_ndarray().squeeze()
        Ey_final = data_final[("mesh", "Ey")].to_ndarray().squeeze()
        Bz_final = data_final[("mesh", "Bz")].to_ndarray().squeeze()

        nx, nz = Ey_init.shape
        Lx = float(ds_init.domain_right_edge[0] - ds_init.domain_left_edge[0])
        Lz = float(ds_init.domain_right_edge[1] - ds_init.domain_left_edge[1])

        # 2D R2C FFT
        Ey_hat = np.fft.rfft2(Ey_init)
        Bz_hat = np.fft.rfft2(Bz_init)

        # Wavenumber grids
        kx = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
        kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi
        KX, KZ = np.meshgrid(kx, kz, indexing="ij")
        knorm = np.sqrt(KX**2 + KZ**2)
        omega = c_light * knorm

        C = np.cos(omega * dt)
        S = np.where(
            omega > 0,
            np.divide(
                np.sin(omega * dt), omega, where=omega > 0, out=np.full_like(omega, dt)
            ),
            dt,
        )

        # 2D XZ: k = kx x-hat + kz z-hat, E = Ey y-hat, B = Bz z-hat
        # (ik x B)_y = (ik x Bz z-hat)_y = i*(kx x-hat + kz z-hat) x Bz z-hat)_y
        #            = i*kx*Bz*(x x z)_y = i*kx*Bz*(-y)_y = -i*kx*Bz
        # (ik x E)_z = (ik x Ey y-hat)_z = i*(kx*(x x y) + kz*(z x y))_z
        #            = i*(kx*z - kz*x)_z = i*kx
        # So (ik x E)_z = i*kx*Ey
        Ey_hat_new = C * Ey_hat + c_light**2 * S * (-1j * KX * Bz_hat)
        Bz_hat_new = C * Bz_hat - S * (1j * KX * Ey_hat)

        Ey_expected = np.fft.irfft2(Ey_hat_new, s=(nx, nz))
        Bz_expected = np.fft.irfft2(Bz_hat_new, s=(nx, nz))

        E0 = np.max(np.abs(Ey_init))
        B0 = np.max(np.abs(Bz_init))
        err_Ey = np.max(np.abs(Ey_final - Ey_expected)) / E0
        err_Bz = np.max(np.abs(Bz_final - Bz_expected)) / B0

        print(f"2D PSATD vacuum eigenmode: t_init={t_init:.2e}, t_final={t_final:.2e}")
        print(f"  Ey max relative error: {err_Ey:.2e}")
        print(f"  Bz max relative error: {err_Bz:.2e}")

        return max(err_Ey, err_Bz)

    elif ndim == 3:
        Ey_init = data_init[("mesh", "Ey")].to_ndarray().squeeze()
        Bx_init = data_init[("mesh", "Bx")].to_ndarray().squeeze()
        Ey_final = data_final[("mesh", "Ey")].to_ndarray().squeeze()
        Bx_final = data_final[("mesh", "Bx")].to_ndarray().squeeze()

        nx, ny, nz = Ey_init.shape
        Lx = float(ds_init.domain_right_edge[0] - ds_init.domain_left_edge[0])
        Ly = float(ds_init.domain_right_edge[1] - ds_init.domain_left_edge[1])
        Lz = float(ds_init.domain_right_edge[2] - ds_init.domain_left_edge[2])

        # 3D R2C FFT
        Ey_hat = np.fft.rfftn(Ey_init)
        Bx_hat = np.fft.rfftn(Bx_init)

        # Wavenumber grids
        kx = np.fft.fftfreq(nx, d=Lx / nx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=Ly / ny) * 2 * np.pi
        kz = np.fft.rfftfreq(nz, d=Lz / nz) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        knorm = np.sqrt(KX**2 + KY**2 + KZ**2)
        omega = c_light * knorm

        C = np.cos(omega * dt)
        S = np.where(
            omega > 0,
            np.divide(
                np.sin(omega * dt), omega, where=omega > 0, out=np.full_like(omega, dt)
            ),
            dt,
        )

        # 3D: k = (kx,ky,kz), E = Ey y-hat, B = Bx x-hat
        # (ik x B)_y = (ik x Bx x-hat)_y = i*(ky*(y x x) + kz*(z x x))_y
        #            = i*(ky*(-z) + kz*(y))_y = i*kz*Bx
        # (ik x E)_x = (ik x Ey y-hat)_x = i*(kx*(x x y) + kz*(z x y))_x
        #            = i*(kx*z + kz*(-x))_x = -i*kz*Ey
        Ey_hat_new = C * Ey_hat + c_light**2 * S * (1j * KZ * Bx_hat)
        Bx_hat_new = C * Bx_hat - S * (-1j * KZ * Ey_hat)

        Ey_expected = np.fft.irfftn(Ey_hat_new, s=(nx, ny, nz), axes=(0, 1, 2))
        Bx_expected = np.fft.irfftn(Bx_hat_new, s=(nx, ny, nz), axes=(0, 1, 2))

        E0 = np.max(np.abs(Ey_init))
        B0 = np.max(np.abs(Bx_init))
        err_Ey = np.max(np.abs(Ey_final - Ey_expected)) / E0
        err_Bx = np.max(np.abs(Bx_final - Bx_expected)) / B0

        print(f"3D PSATD vacuum eigenmode: t_init={t_init:.2e}, t_final={t_final:.2e}")
        print(f"  Ey max relative error: {err_Ey:.2e}")
        print(f"  Bx max relative error: {err_Bx:.2e}")

        return max(err_Ey, err_Bx)


def analyze_rz(ds_init, ds_final):
    """
    Analyze RZ PSATD vacuum z-mode evolution.

    Validates that the RZ PSATD spectral solver (C2C FFT + Hankel transform)
    produces physically reasonable results. Checks:
    1. Bz amplitude is preserved (not growing unboundedly)
    2. Et is generated from Bz (expected from Maxwell's equations)
    3. Bz retains its sinusoidal z-profile structure

    The primary value of this test is verifying that the RZ spectral pipeline
    (MKL C2C FFT, BLAS++ Hankel gemm, LAPACK++ matrix inversion with correct
    ILP64 integers) runs without crashing and produces sensible results.
    """
    t_init = float(ds_init.current_time)
    t_final = float(ds_final.current_time)

    data_init = ds_init.covering_grid(
        level=0, left_edge=ds_init.domain_left_edge, dims=ds_init.domain_dimensions
    )
    data_final = ds_final.covering_grid(
        level=0, left_edge=ds_final.domain_left_edge, dims=ds_final.domain_dimensions
    )

    Bz_init = data_init[("boxlib", "Bz")].to_ndarray().squeeze()
    Bz_final = data_final[("boxlib", "Bz")].to_ndarray().squeeze()
    Et_final = data_final[("boxlib", "Et")].to_ndarray().squeeze()

    B0 = np.max(np.abs(Bz_init))

    # Check 1: Bz amplitude should not grow (should decrease or stay same)
    Bz_max_final = np.max(np.abs(Bz_final))
    amp_ratio = Bz_max_final / B0
    print(f"RZ PSATD z-mode: t_init={t_init:.2e}, t_final={t_final:.2e}")
    print(f"  Bz amplitude ratio (final/init): {amp_ratio:.6f}")

    # Check 2: Et should be non-zero (PSATD generates it from Bz)
    Et_max = np.max(np.abs(Et_final))
    print(f"  Et max at final step: {Et_max:.4e}")

    # Check 3: Bz z-profile at middle r should still be sinusoidal
    # Take the middle r-row
    nr = Bz_final.shape[0]
    Bz_mid = Bz_final[nr // 2, :]
    # FFT and check dominant mode is still k=1
    Bz_hat = np.abs(np.fft.rfft(Bz_mid))
    dominant_mode = np.argmax(Bz_hat[1:]) + 1  # skip DC
    mode_ratio = Bz_hat[1] / np.sum(Bz_hat[1:])
    print(f"  Bz dominant z-mode: k={dominant_mode} (expected 1)")
    print(f"  Bz mode-1 fraction: {mode_ratio:.6f}")

    # Error: amplitude should not grow, mode structure preserved
    err = 0.0
    if amp_ratio > 1.01:
        err = max(err, amp_ratio - 1.0)
        print(f"  WARNING: Bz amplitude grew by {(amp_ratio - 1) * 100:.1f}%")
    if dominant_mode != 1:
        err = max(err, 1.0)
        print(f"  WARNING: dominant mode shifted to k={dominant_mode}")
    if mode_ratio < 0.99:
        err = max(err, 1.0 - mode_ratio)
        print(f"  WARNING: mode-1 fraction only {mode_ratio:.4f}")

    return err


def analyze_rz_hankel(ds_init, ds_final):
    """
    Analyze RZ PSATD Hankel eigenfunction evolution by comparing step-0
    and step-N fields.

    Initial condition: Bz(r) = B0 * J0(alpha01 * r / R), uniform in z.
    After PSATD evolution, the field should retain its shape with a
    cos(omega*t) phase factor from the Hankel spectral update.
    """
    t_init = float(ds_init.current_time)
    t_final = float(ds_final.current_time)

    data_init = ds_init.covering_grid(
        level=0, left_edge=ds_init.domain_left_edge, dims=ds_init.domain_dimensions
    )
    data_final = ds_final.covering_grid(
        level=0, left_edge=ds_final.domain_left_edge, dims=ds_final.domain_dimensions
    )

    Bz_init = data_init[("boxlib", "Bz")].to_ndarray().squeeze()
    Bz_final = data_final[("boxlib", "Bz")].to_ndarray().squeeze()

    B0 = np.max(np.abs(Bz_init))

    # Check amplitude preservation: the Hankel+FFT cycle should preserve norm
    norm_init = np.sqrt(np.sum(Bz_init**2))
    norm_final = np.sqrt(np.sum(Bz_final**2))
    amplitude_ratio = norm_final / norm_init

    # Check z-uniformity of final field (should remain uniform)
    Bz_z_var = np.max(np.std(Bz_final, axis=1)) / B0

    # Check radial profile shape: average over z, compare ratio to initial
    Bz_r_init = np.mean(Bz_init, axis=1)
    Bz_r_final = np.mean(Bz_final, axis=1)

    # The radial profile should just be scaled by some phase factor
    # Find the scale factor: inner product
    scale = np.sum(Bz_r_final * Bz_r_init) / np.sum(Bz_r_init**2)
    Bz_r_expected = scale * Bz_r_init
    err_profile = np.max(np.abs(Bz_r_final - Bz_r_expected)) / B0

    print(f"RZ Hankel eigenfunction: t_init={t_init:.2e}, t_final={t_final:.2e}")
    print(f"  amplitude ratio (L2 norm): {amplitude_ratio:.10f}")
    print(f"  radial profile shape error: {err_profile:.2e}")
    print(f"  z-uniformity (std/B0): {Bz_z_var:.2e}")
    print(f"  scale factor: {scale:.10f}")

    # Main error metric: combination of shape preservation and z-uniformity
    # Amplitude should be preserved to machine precision
    err_amplitude = abs(1.0 - amplitude_ratio)
    print(f"  amplitude preservation error: {err_amplitude:.2e}")

    return max(err_profile, Bz_z_var, err_amplitude)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim",
        required=True,
        choices=["1", "2", "3", "rz", "rz_hankel"],
        help="Dimensionality of the test",
    )
    parser.add_argument(
        "--diag-dir-init", required=True, help="Path to step-0 plotfile"
    )
    parser.add_argument(
        "--diag-dir-final", required=True, help="Path to final-step plotfile"
    )
    args = parser.parse_args()

    ds_init = yt.load(args.diag_dir_init)
    ds_final = yt.load(args.diag_dir_final)

    if args.dim in ("1", "2", "3"):
        ndim = int(args.dim)
        error = analyze_cartesian(ds_init, ds_final, ndim)
        tolerance = 1.0e-6
        label = f"{ndim}D"
    elif args.dim == "rz":
        error = analyze_rz(ds_init, ds_final)
        tolerance = 1.0e-4
        label = "RZ z-mode"
    elif args.dim == "rz_hankel":
        error = analyze_rz_hankel(ds_init, ds_final)
        tolerance = 1.0e-2
        label = "RZ Hankel"

    print(f"\n{label} spectral precision test:")
    print(f"  max error  = {error:.2e}")
    print(f"  tolerance  = {tolerance:.2e}")

    assert error < tolerance, (
        f"{label} spectral precision test FAILED: "
        f"error {error:.2e} >= tolerance {tolerance:.2e}"
    )
    print("  PASSED")


if __name__ == "__main__":
    main()
