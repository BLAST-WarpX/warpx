# Add tests (alphabetical order) ##############################################
#

# Embedded-boundary current FOLD + FILL for an azimuthal (J_theta) band near a
# cylindrical conducting wall: validates the binding against an exact numpy
# replica of the fill operator, the method-of-images reference (convergence
# variant), and the C4 symmetry of the source.
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_eb_image_current_collocated_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_eb_image_current_picmi.py --mode operator --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_eb_image_current_staggered_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_eb_image_current_picmi.py --mode operator --grid-type staggered"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_eb_image_current_convergence_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_eb_image_current_picmi.py --mode convergence --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_embedded_boundary_cylinder_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry cylinder"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_embedded_boundary_marder_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery marder"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_embedded_boundary_plane_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_ohms_law_embedded_boundary_plane_picmi  # name
        2  # dims
        2  # nprocs
        "inputs_test_2d_ohms_law_embedded_boundary_picmi.py --battery eb"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_ohms_law_embedded_boundary_marder_picmi  # name
        2  # dims
        2  # nprocs
        "inputs_test_2d_ohms_law_embedded_boundary_picmi.py --battery marder"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_ohms_law_embedded_boundary_plane_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_ohms_law_embedded_boundary_picmi.py --geometry plane"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_ohms_law_embedded_boundary_cone_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_ohms_law_embedded_boundary_picmi.py --geometry cone"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_hyper_resistivity_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery hyper"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_resistivity_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery resistive"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_ohms_law_resistivity_picmi  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_ohms_law_embedded_boundary_picmi.py --battery resistive"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_ohms_law_hyper_resistivity_picmi  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_ohms_law_embedded_boundary_picmi.py --battery hyper"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_2d_ohms_law_embedded_boundary_divb_clean_picmi  # name
        2  # dims
        1  # nprocs
        "inputs_test_2d_ohms_law_embedded_boundary_picmi.py --battery divb_clean"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_ohms_law_hyper_resistivity_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_ohms_law_embedded_boundary_picmi.py --battery hyper"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

add_warpx_test(
    test_rz_ohms_law_marder_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohms_law_marder_picmi.py"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

# Collocated (nodal) conformal-EB batteries: the masked nodal Faraday update with
# the level-set mirror BC, and the nodal isotropization stencils.
if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_embedded_boundary_collocated_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_marder_collocated_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery marder --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_hyper_resistivity_collocated_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery hyper --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_ohms_law_resistivity_collocated_picmi  # name
        3  # dims
        1  # nprocs
        "inputs_test_3d_ohms_law_embedded_boundary_picmi.py --geometry plane --battery resistive --grid-type collocated"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
