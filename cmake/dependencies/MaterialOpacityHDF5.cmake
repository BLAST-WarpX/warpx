# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

if(WarpX_MATERIAL_OPACITY_HDF5)
    # This dependency is intentionally independent of openPMD and
    # Singularity-EOS.  Only the HDF5 C API is used by the table reader.
    set(_warpx_material_opacity_hdf5_components C)
    if(WarpX_SINGULARITY_EOS)
        # Singularity's Spiner file backend additionally requires HDF5 HL.
        # Request it in the first HDF5 lookup so a combined build does not
        # populate HDF5_LIBRARIES with C only and bypass Singularity's lookup.
        list(APPEND _warpx_material_opacity_hdf5_components HL)
    endif()
    find_package(
        HDF5 1.10 REQUIRED
        COMPONENTS ${_warpx_material_opacity_hdf5_components})
    unset(_warpx_material_opacity_hdf5_components)

    add_library(WarpX_material_opacity_hdf5 INTERFACE)
    add_library(
        WarpX::thirdparty::material_opacity_hdf5
        ALIAS WarpX_material_opacity_hdf5)

    if(TARGET HDF5::HDF5)
        target_link_libraries(
            WarpX_material_opacity_hdf5 INTERFACE HDF5::HDF5)
    else()
        target_include_directories(
            WarpX_material_opacity_hdf5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(
            WarpX_material_opacity_hdf5 INTERFACE ${HDF5_LIBRARIES})
    endif()

    # Some distributions provide only a parallel HDF5 development package.
    # Its public hdf5.h includes mpi.h even when WarpX itself is configured
    # without MPI, so propagate that implementation dependency explicitly.
    if(HDF5_IS_PARALLEL)
        find_package(MPI REQUIRED COMPONENTS C)
        target_link_libraries(
            WarpX_material_opacity_hdf5 INTERFACE MPI::MPI_C)
    endif()
endif()
