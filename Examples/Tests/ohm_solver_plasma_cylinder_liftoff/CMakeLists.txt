# Add tests (alphabetical order) ##############################################
#

# The CI variants pin the particle stack, fill, and stencil choices to the
# values their checksums were blessed with, so they stay independent of the
# deck's production defaults (shape-2/mc-gather and the full isotropized
# operator set incl. the #7040 grad Pe on both grids; esirkepov + cell
# switch + p1w4 blend on staggered; fill 0.20).
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_plasma_cylinder_liftoff_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_plasma_cylinder_liftoff_picmi.py --test --particle-shape 1 --no-mc-gather --fill-frac 0.10 --no-isotropic-gradient"  # inputs
        "analysis.py --path diags/field_diag000010"  # analysis
        "analysis_default_regression.py --path diags/field_diag000010"  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_3d_ohm_solver_plasma_cylinder_liftoff_picmi slow)
endif()

# Staggered (Yee) grid counterpart with the direct level-set mirror wall fill:
# same physics checks as the collocated test, in particular that the external
# drive field fills through the conducting wall (guards the staggered
# curl(A_ext) covered-cell masking regression) and that the early-time
# azimuthal spectrum stays at the particle-noise floor on both grid types.
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohm_solver_plasma_cylinder_liftoff_yee_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohm_solver_plasma_cylinder_liftoff_picmi.py --test --grid-type staggered --no-conformal-eb --eb-b-straight-mirror --particle-shape 1 --deposition direct --no-mc-gather --fill-frac 0.10 --isotropic-resistivity --isotropic-hyper --holmstrom-switch-mode edge --holmstrom-blend-pow 0 --no-isotropic-gradient"  # inputs
        "analysis.py --path diags/field_diag000010"  # analysis
        "analysis_default_regression.py --path diags/field_diag000010"  # checksum
        test_3d_ohm_solver_plasma_cylinder_liftoff_picmi  # dependency
    )
    label_warpx_test(test_3d_ohm_solver_plasma_cylinder_liftoff_yee_picmi slow)
endif()
