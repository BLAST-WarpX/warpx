# Add tests (alphabetical order) ##############################################
#

# Analytic constant-eta magnetic diffusion (Fourier By mode) with
# hybrid_pic_model.implicit_mag_diffusion (matrix-free flat-eta Advance).
# Uses the C++ executable directly (no PICMI) so pywarpx is not required.
add_warpx_test(
    test_2d_hybrid_mag_diffusion  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic comparison in analysis.py)
    OFF  # dependency
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_cn  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_cn"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic comparison in analysis.py)
    OFF  # dependency
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_variable_eta  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_variable_eta"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic comparison in analysis.py)
    OFF  # dependency
)

# Crank-Nicolson (theta=0.5) companion to the 2D variable-eta test. A uniform
# parser eta drives the matrix-free GMRES path with the (1-theta) explicit RHS
# term; the discrete-Fourier CN amplification oracle in analysis.py applies
# unchanged (uniform parser eta matches the constant-eta symbol). Mirrors the
# constant-eta CN analytic test (also matrix-free flat-eta after Phase A).
add_warpx_test(
    test_2d_hybrid_mag_diffusion_variable_eta_cn  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_variable_eta_cn"  # inputs
    "analysis.py"  # analysis
    OFF  # checksum (analytic CN comparison in analysis.py)
    OFF  # dependency
)

# Crank-Nicolson (theta=0.5) smoke with a spatially varying (J-dependent)
# parser-frozen eta on the 2D matrix-free path: exercises the (1-theta) explicit
# RHS term with a genuinely non-uniform eta MultiFab (not a single Fourier mode,
# so analysis_2d_smoke checks finite/damped By rather than the amplification).
add_warpx_test(
    test_2d_hybrid_mag_diffusion_variable_eta_j_cn  # name
    2  # dims
    1  # nprocs
    "inputs_test_2d_hybrid_mag_diffusion_variable_eta_j_cn"  # inputs
    "analysis_2d_smoke.py"  # analysis
    OFF  # checksum (finite/damping smoke in analysis_2d_smoke.py)
    OFF  # dependency
)

add_warpx_test(
    test_rz_hybrid_mag_diffusion_smoke  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_smoke"  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum (finite/damping checks in analysis_rz.py)
    OFF  # dependency
)

# Parser-driven eta on the matrix-free RZ path (uniform resistivity smoke).
# analysis_rz.py checks finite fields and resistive damping, not a Fourier mode.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_variable_eta  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_variable_eta"  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum (finite/damping checks in analysis_rz.py)
    OFF  # dependency
)

# Residual-eta partition: pure-explicit Ohm (ref) vs operator-split with
# mag_diff_eta_explicit_max = eta (so eta_D = max(eta-eta_max,0) must be 0).
# After the residual fix the split run matches the explicit ref; without it
# mag-diff double-counts soft eta and analysis_rz_split_parity.py fails.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_split_parity_ref  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_split_parity_ref"  # inputs
    "analysis_rz.py"  # analysis (finite + damped Bt smoke on the explicit ref)
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_hybrid_mag_diffusion_split_parity  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_split_parity"  # inputs
    "analysis_rz_split_parity.py"  # analysis
    OFF  # checksum (parity oracle vs pure-explicit ref)
    test_rz_hybrid_mag_diffusion_split_parity_ref  # dependency
)

# Crank-Nicolson (theta=0.5) companion to the RZ variable-eta smoke. A uniform
# parser eta (mild 1e-2 Ohm m) drives the matrix-free GMRES path with the
# (1-theta) explicit RHS term; analysis_rz.py checks finite, damped Bt. Mild eta
# (not 129) keeps the CN system well conditioned -- see the stiff-CN limitation
# in inputs_test_rz_hybrid_mag_diffusion_pec_feed_cn / the design note.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_variable_eta_cn  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_variable_eta_cn"  # inputs
    "analysis_rz.py"  # analysis
    OFF  # checksum (finite/damping checks in analysis_rz.py)
    OFF  # dependency
)

# Dirichlet tangential B at the upper-r face via pec_insulator. The feed is
# included in the matrix-free operator (homogeneous part) and RHS (affine
# offset). analysis_rz_pec_feed.py checks that the feed is held and the run
# is stable.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_pec_feed  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_pec_feed"  # inputs
    "analysis_rz_pec_feed.py"  # analysis
    OFF  # checksum (feed checks in analysis_rz_pec_feed.py)
    OFF  # dependency
)

# Crank-Nicolson (theta=0.5) companion to the RZ pec_insulator feed test. The
# matrix-free CN solve keeps the affine feed split (homogeneous operator, feed
# on the RHS) and adds the (1-theta) explicit curl term; the feed is evaluated
# at g(t_new) for both implicit and explicit terms. CN is not L-stable, so with
# stiff vacuum eta=129 it reaches the diffusion equilibrium slowly (oscillating
# stiff modes, g -> -1); analysis_rz_pec_feed_cn.py checks finite/feed-injected/
# axis-regular/stable rather than the theta=1 steady-state "held near B0" check.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_pec_feed_cn  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_pec_feed_cn"  # inputs
    "analysis_rz_pec_feed_cn.py"  # analysis
    OFF  # checksum (stiff-CN feed smoke in analysis_rz_pec_feed_cn.py)
    OFF  # dependency
)

# Non-periodic axial (z) boundaries with PEC endwalls (B_normal = 0,
# tangential mirror). analysis_rz_z_pec_wall.py checks finite damped Bt and
# Bz ~ 0.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_z_pec_wall  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_z_pec_wall"  # inputs
    "analysis_rz_z_pec_wall.py"  # analysis
    OFF  # checksum (PEC endwall checks in analysis_rz_z_pec_wall.py)
    OFF  # dependency
)

# Axial z_lo pec_insulator B_t feed with matching r_hi Dirichlet data so the
# B_t proportional to r mode fills the domain in z (axial coupling).
# analysis_rz_zfeed.py checks deep axial fill of that mode.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_zfeed  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_zfeed"  # inputs
    "analysis_rz_zfeed.py"  # analysis
    OFF  # checksum (axial-fill checks in analysis_rz_zfeed.py)
    OFF  # dependency
)

# Optional PETSc KSP/PC path for the matrix-free variable-eta curl-curl solve
# (mag_diff_linear_solver = petsc, frozen-eta Laplacian Pmat). Same finite/damped
# Bt oracle as the variable-eta smoke; only the linear-solver driver differs.
# Only registered when WarpX is built with PETSc (AMReX_PETSC), mirroring the
# implicit/* curl-curl PETSc PC tests.
if(AMReX_PETSC)
    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_hybrid_mag_diffusion_petsc"  # inputs
        "analysis_rz.py"  # analysis
        OFF  # checksum (RZ vacuum smoke analysis is the oracle)
        OFF  # dependency
    )

    # Crank-Nicolson (theta=0.5) companion to the PETSc KSP smoke, on the 2D
    # matrix-free path with the discrete-Fourier amplification oracle. The
    # theta<1 RHS (2*B^n - A_e(B^n)) is assembled before the GMRES/PETSc branch,
    # so the PETSc KSP inherits it unchanged; mild eta (1e-2) keeps the system
    # well conditioned so PETSc converges tightly and reproduces the CN
    # amplification factor (the matvec is the same homogeneous operator the
    # GMRES path validates). Stiff-eta CN is a known ill-conditioned limitation
    # (see test_rz_hybrid_mag_diffusion_pec_feed_cn and the design note).
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_petsc_cn  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_hybrid_mag_diffusion_petsc_cn"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum (analytic CN comparison in analysis.py)
        OFF  # dependency
    )
endif()

# Stair-case embedded boundary on the RZ matrix-free variable-eta path (θ=1):
# outer-wall solid for r > r_wall, fluid on axis. Covered Bt stays inert (0),
# fluid Bt persists (mild eta). analysis_rz_eb.py checks finite Bt, fluid Bt
# nonzero, damping, and covered-region Bt ~ 0 (global min). EB is a build-time
# option, so only register when WarpX_EB is ON (EB-off CI skips cleanly, as the
# other embedded_boundary_* tests do).
if(WarpX_EB)
    add_warpx_test(
        test_rz_hybrid_mag_diffusion_variable_eta_eb  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_hybrid_mag_diffusion_variable_eta_eb"  # inputs
        "analysis_rz_eb.py"  # analysis
        OFF  # checksum (finite/covered-inert oracle in analysis_rz_eb.py)
        OFF  # dependency
    )
endif()

# ---- RCYLINDER (1D r-only, axisymmetric) matrix-free mag-diff ----------------
# RCYLINDER reuses the RZ cylindrical curl kernels + axis BC, so it routes
# through the matrix-free AdvanceVariable path.
# r_hi = PEC (Neumann dBt/dr=0) is REQUIRED for a well-posed Bt solve: with
# "none" at r_hi the cylindrical operator has a Bt proportional-to-r steady-
# state null mode (K(Bt=r)=0) leaving the system singular, so AMReX GMRES and
# PETSc converge to different particular solutions. PEC excludes Bt=r
# (dBt/dr=k!=0) -> unique solution (verified AMReX==PETSc to ~1e-10). See
# notes/2026-07-19_rcyl_mag_diff_port.md. theta=1 only.

# T1: harsh vacuum (eta=129, dt=1ns) AMReX GMRES. Oracle (analysis_rcyl.py):
# finite Bt; >99.9% damped in 5 steps (slowest Bessel mode annihilated); AND a
# paired analytic explicit-resistive CFL-violation analysis (const_dt /
# dt_resistive_explicit ~ 2e6 -> explicit Faraday wildly unstable at this eta;
# the operator-split implicit mag-diff is what makes the deck run).
add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_harsh  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_hybrid_mag_diffusion_harsh"  # inputs
    "analysis_rcyl.py"  # analysis
    OFF  # checksum (finite + >99.9% damped + explicit-CFL analysis in analysis_rcyl.py)
    OFF  # dependency
)

# T2 reference: mild eta (1e-2) AMReX GMRES. Persists a nonzero Bt (slowest mode
# damps only ~0.03%/step) so the field is meaningful for the PETSc parity L2.
# analysis_rcyl_mild.py checks finite + damped; the plotfile + reduced
# field_energy are the parity reference for test_rcylinder_hybrid_mag_diffusion_petsc.
add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_mild  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_hybrid_mag_diffusion_mild"  # inputs
    "analysis_rcyl_mild.py"  # analysis
    OFF  # checksum (finite + damped smoke in analysis_rcyl_mild.py)
    OFF  # dependency
)

# T2 + T3: PETSc (exact_curl_curl) solver parity to the AMReX GMRES reference.
# IDENTICAL physics to test_rcylinder_hybrid_mag_diffusion_mild, differing only in
# mag_diff_linear_solver. analysis_rcyl_parity.py compares this run's reduced
# B-energy + Bt Cell_H min/max to the sibling reference test's (a ctest
# `dependency` keeps the reference dir alive through this test). Manual yt
# full-field L2 ~1e-10; the ctest uses stdlib-readable reduced B-energy + Cell_H
# (the test python has no yt/numpy). Only registered with PETSc (mirrors the RZ
# PETSc test gating).
if(AMReX_PETSC)
    add_warpx_test(
        test_rcylinder_hybrid_mag_diffusion_petsc  # name
        RCYLINDER  # dims
        1  # nprocs
        "inputs_test_rcylinder_hybrid_mag_diffusion_petsc"  # inputs
        "analysis_rcyl_parity.py"  # analysis
        OFF  # checksum (solver-parity oracle in analysis_rcyl_parity.py)
        test_rcylinder_hybrid_mag_diffusion_mild  # dependency (AMReX GMRES reference)
    )

    # T2: harsh vacuum (eta=129, dt=1ns, PEC r_hi, 5 steps) through the PETSc
    # KSP + cylindrical exact_curl_curl Pmat. Same finite/>99.9%-damped oracle
    # as the AMReX GMRES harsh test (analysis_rcyl.py, reused) -- a parity L2 of
    # the annihilated ~1e-7 Bt fields would be too noisy, so the absolute damping
    # oracle is the strong check. The metric-aware Pmat is what makes the frozen
    # path usable at stiff eta=129 (the Cartesian Pmat was never exercised here).
    add_warpx_test(
        test_rcylinder_hybrid_mag_diffusion_harsh_petsc  # name
        RCYLINDER  # dims
        1  # nprocs
        "inputs_test_rcylinder_hybrid_mag_diffusion_harsh_petsc"  # inputs
        "analysis_rcyl.py"  # analysis (finite + >99.9% damped + explicit-CFL)
        OFF  # checksum (absolute damping oracle in analysis_rcyl.py)
        OFF  # dependency (absolute oracle; no reference dir needed)
    )
endif()
