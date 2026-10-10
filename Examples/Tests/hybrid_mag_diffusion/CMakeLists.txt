# Add tests (alphabetical order) ##############################################

#
# Coverage is intentionally factorized rather than taking the Cartesian product
# of eta source, theta value, linear solver, and geometry. The theta RHS is
# built before solver dispatch, while the PETSc tests focus on exact-Pmat and
# AMReX parity. Add combination tests only for interactions with distinct logic
# (variable coefficients, affine feeds, EB, or geometry-specific boundaries).

# Cartesian 1D_Z physical-boundary regression: a homogeneous PEC endwall and
# an inhomogeneous pec_insulator tangential-B feed exercise both boundary
# contributions in the matrix-free operator.
add_warpx_test(
    test_1d_hybrid_mag_diffusion_pec_feed  # name
    1  # dims
    1  # nprocs
    "inputs_test_1d_hybrid_mag_diffusion_pec_feed"  # inputs
    "analysis_cartesian_boundaries_eb.py"  # analysis
    OFF  # checksum (PEC/feed checks in analysis)
    OFF  # dependency
)

# Analytic constant-eta magnetic diffusion (Fourier By mode). Python-enabled
# builds use HybridPICSolver to cover PICMI input emission; other builds retain
# the native deck, so this core regression is never lost and is not duplicated.
if(WarpX_PYTHON)
    add_warpx_test(
        test_2d_hybrid_mag_diffusion  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_hybrid_mag_diffusion_picmi.py"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum (analytic comparison in analysis.py)
        OFF  # dependency
    )
else()
    add_warpx_test(
        test_2d_hybrid_mag_diffusion  # name
        2  # dims
        1  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion"  # inputs
        "analysis.py"  # analysis
        OFF  # checksum (analytic comparison in analysis.py)
        OFF  # dependency
    )
endif()

# Crank-Nicolson (theta=0.5) with a uniform parser eta. This combines the
# quantitative variable-eta and theta-method coverage in one analytic test;
# the constant-eta backward-Euler test above covers the other eta source.
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

# Smooth constant-eta mode with both resistivity branches active. The explicit
# reference advances all eta through Ohm; the split run advances 40% explicitly
# and 60% implicitly, then compares every magnetic-field cell.
add_warpx_test(
    test_2d_hybrid_mag_diffusion_split_partial_ref  # name
    2  # dims
    1  # nprocs
    "inputs_base_2d_hybrid_mag_diffusion hybrid_pic_model.implicit_mag_diffusion=0 hybrid_pic_model.plasma_resistivity(rho,J,t)=1.0e-2"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_split_partial  # name
    2  # dims
    1  # nprocs
    "inputs_base_2d_hybrid_mag_diffusion hybrid_pic_model.plasma_resistivity(rho,J,t)=1.0e-2 hybrid_pic_model.mag_diff_eta_explicit_max=4.0e-3"  # inputs
    "analysis_rz_split_parity.py ../test_2d_hybrid_mag_diffusion_split_partial_ref require_partial"  # analysis
    OFF  # checksum
    test_2d_hybrid_mag_diffusion_split_partial_ref  # dependency
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

# Residual-eta partition: pure-explicit Ohm (ref) vs operator-split with
# mag_diff_eta_explicit_max = eta (so eta_D = max(eta-eta_max,0) must be 0).
# After the residual fix the split run matches the explicit ref; without it
# mag-diff double-counts soft eta and analysis_rz_split_parity.py fails.
add_warpx_test(
    test_rz_hybrid_mag_diffusion_split_parity_ref  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_hybrid_mag_diffusion_split_parity_ref"  # inputs
    OFF  # analysis (the dependent parity analysis validates this reference)
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
# (mag_diff_linear_solver = petsc, assembled frozen-eta curl-curl Pmat). Same
# finite/damped Bt oracle as the variable-eta smoke; only the linear-solver
# driver differs.
# Only registered when WarpX is built with PETSc (AMReX_PETSC), mirroring the
# implicit/* curl-curl PETSc PC tests.
if(AMReX_PETSC)
    # Cartesian 1D_Z exact-Pmat regression. Variable eta diffuses both
    # transverse components, while longitudinal Bz checks the identity block.
    # Direct LU must reduce the MatShell solve to one iteration, and the
    # analysis compares the result with the AMReX GMRES reference.
    add_warpx_test(
        test_1d_hybrid_mag_diffusion_coupled  # name
        1  # dims
        1  # nprocs
        "inputs_test_1d_hybrid_mag_diffusion_coupled"  # inputs
        OFF  # analysis (reference data consumed by PETSc companion)
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_1d_hybrid_mag_diffusion_coupled_petsc  # name
        1  # dims
        1  # nprocs
        "inputs_test_1d_hybrid_mag_diffusion_coupled_petsc"  # inputs
        "analysis_cartesian_coupled_parity.py"  # analysis
        OFF  # checksum (solver parity + exact-Pmat iteration oracle)
        test_1d_hybrid_mag_diffusion_coupled  # dependency (AMReX reference)
    )
    if(TEST test_1d_hybrid_mag_diffusion_coupled_petsc.run)
        set_property(
            TEST test_1d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY PASS_REGULAR_EXPRESSION "PETSc KSP iterations=1"
        )
        set_property(
            TEST test_1d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY FAIL_REGULAR_EXPRESSION
            "PETSc KSP iterations=([2-9]|[1-9][0-9]);did not converge"
        )
    endif()

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_hybrid_mag_diffusion_petsc"  # inputs
        "analysis_rz.py"  # analysis
        OFF  # checksum (RZ vacuum smoke analysis is the oracle)
        OFF  # dependency
    )

    # A periodic smoke does not exercise clamped normal-B boundary rows.
    # Check the native and assembled operators with an independent mixed-field
    # vector, then require a finite, damped physical trajectory on two ranks.
    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc_boundary_default_pec_insulator
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_petsc_boundary"
        "analysis_petsc_boundary.py"
        OFF
        OFF
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc_boundary_pec
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_petsc_boundary boundary.field_lo=none pec boundary.field_hi=pec pec"
        "analysis_petsc_boundary.py"
        OFF
        OFF
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc_boundary_evolved_insulator
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_petsc_boundary insulator.area_x_hi(y,z)=1 insulator.area_z_lo(x,y)=1 insulator.area_z_hi(x,y)=1"
        "analysis_petsc_boundary.py evolved"
        OFF
        OFF
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc_boundary_forced_feed
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_petsc_boundary insulator.area_x_hi(y,z)=1 insulator.By_x_hi(y,z,t)=1.e-4 warpx.B_external_grid=0 0 0"
        "analysis_petsc_boundary.py feed"
        OFF
        OFF
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_petsc_boundary_held_guards
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_petsc_boundary boundary.field_hi=none pec_insulator"
        "analysis_petsc_boundary.py"
        OFF
        OFF
    )

    foreach(boundary_case IN ITEMS default_pec_insulator pec evolved_insulator forced_feed held_guards)
        set(boundary_test "test_rz_hybrid_mag_diffusion_petsc_boundary_${boundary_case}")
        if(TEST ${boundary_test}.run)
            set_property(
                TEST ${boundary_test}.run APPEND PROPERTY ENVIRONMENT
                "PETSC_OPTIONS=-magdiff_check_finite -magdiff_audit_matrix -magdiff_audit_file pmat_audit.json"
            )
        endif()
    endforeach()

    # Cartesian XZ exact-Pmat regression. The reference and PETSc runs use a
    # divergence-free coupled Bx/Bz mode and spatially varying eta. With a
    # direct LU factorization of P, an exact Pmat must converge the MatShell
    # system in one iteration. The analysis also compares all field extrema and
    # magnetic energy against the AMReX GMRES reference.
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_coupled  # name
        2  # dims
        1  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_coupled"  # inputs
        OFF  # analysis (reference data consumed by PETSc companion)
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_coupled_petsc  # name
        2  # dims
        1  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_coupled_petsc"  # inputs
        "analysis_cartesian_coupled_parity.py"  # analysis
        OFF  # checksum (solver parity + exact-Pmat iteration oracle)
        test_2d_hybrid_mag_diffusion_coupled  # dependency (AMReX reference)
    )
    if(TEST test_2d_hybrid_mag_diffusion_coupled_petsc.run)
        set_property(
            TEST test_2d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY PASS_REGULAR_EXPRESSION "PETSc KSP iterations=1"
        )
        set_property(
            TEST test_2d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY FAIL_REGULAR_EXPRESSION
            "PETSc KSP iterations=([2-9]|[1-9][0-9]);did not converge"
        )
    endif()

    # Two ranks and four boxes exercise PETSc ownership, off-process columns,
    # ghosted indices, ASM overlap, and the independently distributed AMReX
    # matrix-free path. The dependent analysis compares every field cell.
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_coupled_mpi  # name
        2  # dims
        2  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_coupled amr.max_grid_size=16"  # inputs
        "analysis_cartesian_coupled_parity.py ../test_2d_hybrid_mag_diffusion_coupled"  # analysis
        OFF  # checksum
        test_2d_hybrid_mag_diffusion_coupled  # dependency (one-rank reference)
    )
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_coupled_mpi_petsc  # name
        2  # dims
        2  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_coupled_petsc amr.max_grid_size=16 hybrid_pic_model.mag_diff_petsc_pc_type=asm hybrid_pic_model.mag_diff_petsc_asm_overlap=1 hybrid_pic_model.mag_diff_petsc_sub_ksp_type=preonly hybrid_pic_model.mag_diff_petsc_sub_pc_type=ilu hybrid_pic_model.mag_diff_petsc_ilu_factor_levels=0"  # inputs
        "analysis_cartesian_coupled_parity.py"  # analysis
        OFF  # checksum
        test_2d_hybrid_mag_diffusion_coupled_mpi  # dependency
    )

    add_warpx_test(
        test_2d_hybrid_mag_diffusion_nonconvergence  # name
        2  # dims
        2  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_coupled_petsc max_step=1 amr.max_grid_size=16 hybrid_pic_model.mag_diff_max_iter=1 hybrid_pic_model.mag_diff_petsc_pc_type=asm hybrid_pic_model.mag_diff_petsc_sub_pc_type=ilu"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    if(TEST test_2d_hybrid_mag_diffusion_nonconvergence.run)
        set_property(
            TEST test_2d_hybrid_mag_diffusion_nonconvergence.run
            PROPERTY WILL_FAIL TRUE
        )
    endif()

    # Cartesian 3D exact-Pmat regression. A divergence-free three-component B
    # and variable eta exercise all same-component and mixed-derivative blocks.
    add_warpx_test(
        test_3d_hybrid_mag_diffusion_coupled  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_hybrid_mag_diffusion_coupled"  # inputs
        OFF  # analysis (reference data consumed by PETSc companion)
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_3d_hybrid_mag_diffusion_coupled_petsc  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_hybrid_mag_diffusion_coupled_petsc"  # inputs
        "analysis_cartesian_coupled_parity.py"  # analysis
        OFF  # checksum (solver parity + exact-Pmat iteration oracle)
        test_3d_hybrid_mag_diffusion_coupled  # dependency (AMReX reference)
    )
    if(TEST test_3d_hybrid_mag_diffusion_coupled_petsc.run)
        set_property(
            TEST test_3d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY PASS_REGULAR_EXPRESSION "PETSc KSP iterations=1"
        )
        set_property(
            TEST test_3d_hybrid_mag_diffusion_coupled_petsc.run
            PROPERTY FAIL_REGULAR_EXPRESSION
            "PETSc KSP iterations=([2-9]|[1-9][0-9]);did not converge"
        )
    endif()
endif()

# RZ matrix-free variable-eta EB test. The analysis inspects fluid and covered
# regions separately. Register only when EB support is enabled.
if(WarpX_EB)
    # XZ and 3D each combine a PEC wall, a pec_insulator feed, and an interior
    # stair-step EB. This proves the interactions without multiplying separate
    # smoke tests for each feature and dimension.
    add_warpx_test(
        test_2d_hybrid_mag_diffusion_boundaries_eb  # name
        2  # dims
        1  # nprocs
        "inputs_base_2d_hybrid_mag_diffusion_boundaries_eb"  # inputs
        "analysis_cartesian_boundaries_eb.py"  # analysis
        OFF  # checksum (PEC/feed/covered-field checks in analysis)
        OFF  # dependency
    )

    if(AMReX_PETSC)
        add_warpx_test(
            test_2d_hybrid_mag_diffusion_boundaries_eb_mpi_petsc  # name
            2  # dims
            2  # nprocs
            "inputs_base_2d_hybrid_mag_diffusion_boundaries_eb amr.max_grid_size=16 hybrid_pic_model.mag_diff_linear_solver=petsc hybrid_pic_model.mag_diff_petsc_pc_type=asm hybrid_pic_model.mag_diff_petsc_asm_overlap=1 hybrid_pic_model.mag_diff_petsc_sub_ksp_type=preonly hybrid_pic_model.mag_diff_petsc_sub_pc_type=ilu hybrid_pic_model.mag_diff_petsc_ilu_factor_levels=0"  # inputs
            "analysis_cartesian_boundaries_eb.py"  # analysis
            OFF  # checksum
            OFF  # dependency
        )
    endif()

    add_warpx_test(
        test_3d_hybrid_mag_diffusion_boundaries_eb  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_hybrid_mag_diffusion_boundaries_eb"  # inputs
        "analysis_cartesian_boundaries_eb.py"  # analysis
        OFF  # checksum (PEC/feed/covered-field checks in analysis)
        OFF  # dependency
    )

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

# RCYLINDER uses the cylindrical matrix-free operator. A PEC outer boundary
# excludes the Bt proportional-to-r null mode and makes the solve unique.
add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_harsh  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_hybrid_mag_diffusion_harsh"  # inputs
    "analysis_rcyl.py"  # analysis
    OFF  # checksum (finite + >99.9% damped + explicit-CFL analysis in analysis_rcyl.py)
    OFF  # dependency
)

# Mild-eta AMReX reference for PETSc full-field parity.
add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_mild  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_hybrid_mag_diffusion_mild"  # inputs
    "analysis_rcyl_mild.py"  # analysis
    OFF  # checksum (finite + damped smoke in analysis_rcyl_mild.py)
    OFF  # dependency
)

# Variable eta with a nonzero pec_insulator B_theta feed. This covers the
# parser-built eta path and affine boundary feed together in RCYLINDER.
add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_variable_eta_feed  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_test_rcylinder_hybrid_mag_diffusion_variable_eta_feed"  # inputs
    "analysis_rz_pec_feed.py"  # geometry-independent Bt feed oracle
    OFF  # checksum
    OFF  # dependency
)

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

    # Stiff-eta PETSc path with the metric-aware cylindrical Pmat.
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

# Keep literal names and base-deck names visible to the static CI input checker.
# All native controls compare complete fields with existing physical references.
add_warpx_test(
    test_1d_hybrid_mag_diffusion_native_bicgstab
    1
    2
    "inputs_base_1d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_1d_hybrid_mag_diffusion_coupled cartesian"
    OFF
    test_1d_hybrid_mag_diffusion_coupled
)

add_warpx_test(
    test_2d_hybrid_mag_diffusion_native_bicgstab
    2
    2
    "inputs_base_2d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_2d_hybrid_mag_diffusion_coupled cartesian"
    OFF
    test_2d_hybrid_mag_diffusion_coupled
)

add_warpx_test(
    test_3d_hybrid_mag_diffusion_native_bicgstab
    3
    2
    "inputs_base_3d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_3d_hybrid_mag_diffusion_coupled cartesian"
    OFF
    test_3d_hybrid_mag_diffusion_coupled
)

add_warpx_test(
    test_rz_hybrid_mag_diffusion_native_bicgstab
    RZ
    2
    "inputs_base_rz_hybrid_mag_diffusion_smoke hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_rz_hybrid_mag_diffusion_smoke cylindrical"
    OFF
    test_rz_hybrid_mag_diffusion_smoke
)

add_warpx_test(
    test_rcylinder_hybrid_mag_diffusion_native_bicgstab
    RCYLINDER
    2
    "inputs_base_rcylinder_hybrid_mag_diffusion_mild hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_rcylinder_hybrid_mag_diffusion_mild cylindrical"
    OFF
    test_rcylinder_hybrid_mag_diffusion_mild
)

if(AMReX_PETSC)
    add_warpx_test(
        test_1d_hybrid_mag_diffusion_native_ilu_amrex_gmres
        1
        2
        "inputs_base_1d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_gmres hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_1d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_1d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_1d_hybrid_mag_diffusion_native_ilu_amrex_bicgstab
        1
        2
        "inputs_base_1d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_1d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_1d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_2d_hybrid_mag_diffusion_native_ilu_amrex_gmres
        2
        2
        "inputs_base_2d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_gmres hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_2d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_2d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_2d_hybrid_mag_diffusion_native_ilu_amrex_bicgstab
        2
        2
        "inputs_base_2d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_2d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_2d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_3d_hybrid_mag_diffusion_native_ilu_amrex_gmres
        3
        2
        "inputs_base_3d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_gmres hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_3d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_3d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_3d_hybrid_mag_diffusion_native_ilu_amrex_bicgstab
        3
        2
        "inputs_base_3d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_3d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_3d_hybrid_mag_diffusion_coupled
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_native_ilu_amrex_gmres
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_smoke hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_gmres hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_rz_hybrid_mag_diffusion_smoke cylindrical"
        OFF
        test_rz_hybrid_mag_diffusion_smoke
    )

    add_warpx_test(
        test_rz_hybrid_mag_diffusion_native_ilu_amrex_bicgstab
        RZ
        2
        "inputs_base_rz_hybrid_mag_diffusion_smoke hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_rz_hybrid_mag_diffusion_smoke cylindrical"
        OFF
        test_rz_hybrid_mag_diffusion_smoke
    )

    add_warpx_test(
        test_rcylinder_hybrid_mag_diffusion_native_ilu_amrex_gmres
        RCYLINDER
        2
        "inputs_base_rcylinder_hybrid_mag_diffusion_mild hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_gmres hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_rcylinder_hybrid_mag_diffusion_mild cylindrical"
        OFF
        test_rcylinder_hybrid_mag_diffusion_mild
    )

    add_warpx_test(
        test_rcylinder_hybrid_mag_diffusion_native_ilu_amrex_bicgstab
        RCYLINDER
        2
        "inputs_base_rcylinder_hybrid_mag_diffusion_mild hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_residual_replacement_interval=2 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_rcylinder_hybrid_mag_diffusion_mild cylindrical"
        OFF
        test_rcylinder_hybrid_mag_diffusion_mild
    )

    # One grid on two ranks exercises the native ILU empty-rank path.
    add_warpx_test(
        test_1d_hybrid_mag_diffusion_native_ilu_empty_rank
        1
        2
        "inputs_base_1d_hybrid_mag_diffusion_coupled amr.max_grid_size=32 hybrid_pic_model.mag_diff_native_ilu=1 hybrid_pic_model.mag_diff_verbose=1"
        "analysis_native_solver_parity.py ../test_1d_hybrid_mag_diffusion_coupled cartesian"
        OFF
        test_1d_hybrid_mag_diffusion_coupled
    )

endif()

# Force early fallback and independently compare the completed fields.
add_warpx_test(
    test_2d_hybrid_mag_diffusion_native_bicgstab_fallback
    2
    2
    "inputs_base_2d_hybrid_mag_diffusion_coupled hybrid_pic_model.mag_diff_linear_solver=amrex_bicgstab hybrid_pic_model.mag_diff_bicgstab_max_iter=1 hybrid_pic_model.mag_diff_verbose=1"
    "analysis_native_solver_parity.py ../test_2d_hybrid_mag_diffusion_coupled cartesian"
    OFF
    test_2d_hybrid_mag_diffusion_coupled
)
