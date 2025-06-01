# WarpX CMake 配置文件
# 提供组件化安装支持 (app/lib/eb) 和依赖管理
# only add PUBLIC dependencies as well
# 仅添加 PUBLIC 依赖项
#   https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#creating-a-package-configuration-file
include(CMakeFindDependencyMacro)

# Search in <PackageName>_ROOT:
# 在 <PackageName>_ROOT 中搜索：
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
# 启用新版本策略
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# locate the installed FindABC.cmake module for ABC
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules")
# 为ABC查找已安装的FindABC.cmake模块list(APPEND CMAKE_MODULE_PATH “${CMAKE_CURRENT_LIST_DIR}/Modules”)
set(WarpX_APP @WarpX_APP@)
set(WarpX_APP_FOUND ${WarpX_APP})
set(WarpX_EB @WarpX_EB@)
set(WarpX_EB_FOUND ${WarpX_EB})
set(WarpX_LIB @WarpX_LIB@)
set(WarpX_LIB_FOUND ${WarpX_LIB})

# dependencies依赖项配置
# MPI 支持
set(WarpX_MPI @WarpX_MPI@)
if(WarpX_MPI)
    find_dependency(MPI)
    # deselect parallel installs if explicitly a serial install is requested
    # 如果明确要求串行安装，则取消选择并行安装
    # 标记串行版本不可用
    set(WarpX_NOMPI_FOUND FALSE)
else()
    # 标记串行版本可用
    set(WarpX_NOMPI_FOUND TRUE)
endif()
set(WarpX_MPI_FOUND ${WarpX_MPI})

# openPMD 支持
set(WarpX_OPENPMD @WarpX_OPENPMD@)
if(WarpX_OPENPMD)
    find_dependency(openPMD)
endif()
set(WarpX_OPENPMD_FOUND ${WarpX_OPENPMD})

# TODO 未来
#WarpX_ASCENT        "Ascent in situ diagnostics"                 OFF)
#WarpX_CATALYST      "Catalyst in situ diagnostics"               OFF)
#WarpX_FFT           "FFT-based solvers"                          OFF)
#WarpX_SENSEI        "SENSEI in situ diagnostics"                 OFF)
#WarpX_QED           "QED support (requires PICSAR)"                    ON)
#WarpX_QED_TABLE_GEN "QED table generation (requires PICSAR and Boost)" OFF)


# ADIOS2 支持
set(WarpX_HAVE_ADIOS2 @WarpX_HAVE_ADIOS2@)
if(WarpX_HAVE_ADIOS2)
    find_dependency(ADIOS2)
endif()
set(WarpX_ADIOS2_FOUND ${WarpX_HAVE_ADIOS2})

# define central WarpX::app WarpX::shared ... targets
#ABLASTR here, too? or separate Config.cmake?
# 定义中央 WarpX::app WarpX::shared ......这里也以 ABLASTR 为目标？还是单独的 Config.cmake？
# 加载目标文件
include("${CMAKE_CURRENT_LIST_DIR}/WarpXTargets.cmake")

# check if components are fulfilled and set WarpX_<COMPONENT>_FOUND vars
# 检查组件是否满足要求，并设置 WarpX_<COMPONENT>_FOUND vars，
foreach(comp ${WarpX_FIND_COMPONENTS})
    if(NOT WarpX_${comp}_FOUND)
        if(WarpX_FIND_REQUIRED_${comp})
            set(WarpX_FOUND FALSE)
        endif()
    endif()
endforeach()
