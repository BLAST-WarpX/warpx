# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_qed_breit_wheeler  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_qed_breit_wheeler  # inputs
    "analysis_breit_wheeler_yt.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_qed_breit_wheeler_opmd  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_qed_breit_wheeler_opmd  # inputs
    "analysis_breit_wheeler_opmd.py diags/diag1/"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_qed_quantum_sync  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_qed_quantum_sync  # inputs
    "analysis_quantum_sync.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_breit_wheeler  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_breit_wheeler  # inputs
    "analysis_breit_wheeler_yt.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_breit_wheeler_opmd  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_breit_wheeler_opmd  # inputs
    "analysis_breit_wheeler_opmd.py diags/diag1/"  # analysis
    "analysis_default_regression.py --path diags/diag1/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_quantum_sync  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_quantum_sync  # inputs
    "analysis_quantum_sync.py diags/diag1000002"  # analysis
    "analysis_default_regression.py --path diags/diag1000002"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_schwinger_1  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_schwinger_1  # inputs
    "analysis_schwinger.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_schwinger_2  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_schwinger_2  # inputs
    "analysis_schwinger.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_schwinger_3  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_schwinger_3  # inputs
    "analysis_schwinger.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_qed_schwinger_4  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_qed_schwinger_4  # inputs
    "analysis_schwinger.py diags/diag1000001"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
