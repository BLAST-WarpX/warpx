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
