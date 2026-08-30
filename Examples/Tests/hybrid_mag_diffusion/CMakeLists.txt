# Add tests (alphabetical order) ##############################################
#

# Analytic constant-eta magnetic diffusion (Fourier By mode) with
# hybrid_pic_model.implicit_mag_diffusion. Uses the C++ executable directly
# (no PICMI) so pywarpx is not required.
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
# unchanged (uniform parser eta matches the constant-eta symbol). This mirrors
# the constant-eta MLCurlCurl CN test on the matrix-free path.
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
