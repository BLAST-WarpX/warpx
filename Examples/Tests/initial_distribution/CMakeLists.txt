# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_initial_distribution_prepare  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution_prepare.py  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_initial_distribution  # name
    3  # dims
    1  # nprocs
    inputs_test_3d_initial_distribution  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    test_3d_initial_distribution_prepare  # dependency
)

add_warpx_test(
    test_rcylinder_temperature_from_file_prepare  # name
    RCYLINDER  # dims
    1  # nprocs
    "inputs_base_temperature_from_file_prepare.py --geometry rcylinder"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rcylinder_temperature_from_file  # name
    RCYLINDER  # dims
    1  # nprocs
    inputs_test_rcylinder_temperature_from_file  # inputs
    "analysis_temperature_from_file.py --geometry rcylinder"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    test_rcylinder_temperature_from_file_prepare  # dependency
)

add_warpx_test(
    test_rsphere_temperature_from_file_prepare  # name
    RSPHERE  # dims
    1  # nprocs
    "inputs_base_temperature_from_file_prepare.py --geometry rsphere"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rsphere_temperature_from_file  # name
    RSPHERE  # dims
    1  # nprocs
    inputs_test_rsphere_temperature_from_file  # inputs
    "analysis_temperature_from_file.py --geometry rsphere"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    test_rsphere_temperature_from_file_prepare  # dependency
)

add_warpx_test(
    test_rz_temperature_from_file_prepare  # name
    RZ  # dims
    1  # nprocs
    "inputs_base_temperature_from_file_prepare.py --geometry rz"  # inputs
    OFF  # analysis
    OFF  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_temperature_from_file  # name
    RZ  # dims
    1  # nprocs
    inputs_test_rz_temperature_from_file  # inputs
    "analysis_temperature_from_file.py --geometry rz"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    test_rz_temperature_from_file_prepare  # dependency
)
