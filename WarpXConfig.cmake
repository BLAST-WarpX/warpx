# only add PUBLIC dependencies as well
#   https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#creating-a-package-configuration-file
include(CMakeFindDependencyMacro)

# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# locate the installed FindABC.cmake module for ABC
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules")

set(WarpX_APP @WarpX_APP@)
set(WarpX_APP_FOUND ${WarpX_APP})
set(WarpX_EB @WarpX_EB@)
set(WarpX_EB_FOUND ${WarpX_EB})
set(WarpX_LIB @WarpX_LIB@)
set(WarpX_LIB_FOUND ${WarpX_LIB})

# dependencies
set(WarpX_MPI @WarpX_MPI@)
if(WarpX_MPI)
    find_dependency(MPI)
    # deselect parallel installs if explicitly a serial install is requested
    set(WarpX_NOMPI_FOUND FALSE)
else()
    set(WarpX_NOMPI_FOUND TRUE)
endif()
set(WarpX_MPI_FOUND ${WarpX_MPI})

set(WarpX_OPENPMD @WarpX_OPENPMD@)
if(WarpX_OPENPMD)
    find_dependency(openPMD)
endif()
set(WarpX_OPENPMD_FOUND ${WarpX_OPENPMD})

# TODO
#WarpX_ASCENT        "Ascent in situ diagnostics"                 OFF)
#WarpX_CATALYST      "Catalyst in situ diagnostics"               OFF)
#WarpX_FFT           "FFT-based solvers"                          OFF)
#WarpX_SENSEI        "SENSEI in situ diagnostics"                 OFF)
#WarpX_QED           "QED support (requires PICSAR)"                    ON)
#WarpX_QED_TABLE_GEN "QED table generation (requires PICSAR and Boost)" OFF)



set(WarpX_HAVE_ADIOS2 @WarpX_HAVE_ADIOS2@)
if(WarpX_HAVE_ADIOS2)
    find_dependency(ADIOS2)
endif()
set(WarpX_ADIOS2_FOUND ${WarpX_HAVE_ADIOS2})

# define central WarpX::app WarpX::shared ... targets
#ABLASTR here, too? or separate Config.cmake?
include("${CMAKE_CURRENT_LIST_DIR}/WarpXTargets.cmake")

# check if components are fulfilled and set WarpX_<COMPONENT>_FOUND vars
foreach(comp ${WarpX_FIND_COMPONENTS})
    if(NOT WarpX_${comp}_FOUND)
        if(WarpX_FIND_REQUIRED_${comp})
            set(WarpX_FOUND FALSE)
        endif()
    endif()
endforeach()
