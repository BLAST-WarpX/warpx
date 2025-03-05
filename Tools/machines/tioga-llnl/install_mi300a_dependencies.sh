#!/bin/bash
#
# Copyright 2024 The WarpX Community
#
# This file is part of WarpX.
#
# Author: Axel Huebl
# License: BSD-3-Clause-LBNL

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was tioga_mi300a_warpx.profile sourced and configured correctly?
#   early access: not yet used!
#if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your tioga_mi300a_warpx.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SRC_DIR="/p/lustre1/${USER}/tioga/src"
SW_DIR="/p/lustre1/${USER}/tioga/warpx/mi300a"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y pywarpx
python3 -m pip uninstall -qq -y warpx
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# General extra dependencies ##################################################
#

# tmpfs build directory: avoids issues often seen with $HOME and is faster
build_dir=$(mktemp -d)
build_procs=24

# C-Blosc2 (I/O compression)
if [ -d ${SRC_DIR}/c-blosc2 ]
then
  cd ${SRC_DIR}/c-blosc2
  git fetch --prune
  git checkout v2.15.1
  cd -
else
  git clone -b v2.15.1 https://github.com/Blosc/c-blosc2.git ${SRC_DIR}/c-blosc2
fi
cmake \
    --fresh                        \
    -S ${SRC_DIR}/c-blosc2         \
    -B ${build_dir}/c-blosc2-build \
    -DBUILD_TESTS=OFF              \
    -DBUILD_BENCHMARKS=OFF         \
    -DBUILD_EXAMPLES=OFF           \
    -DBUILD_FUZZERS=OFF            \
    -DBUILD_STATIC=OFF             \
    -DDEACTIVATE_AVX2=OFF          \
    -DDEACTIVATE_AVX512=OFF        \
    -DWITH_SANITIZER=OFF           \
    -DCMAKE_INSTALL_PREFIX=${SW_DIR}/c-blosc-2.15.1
cmake \
    --build ${build_dir}/c-blosc2-build \
    --target install                    \
    --parallel ${build_procs}
rm -rf ${build_dir}/c-blosc2-build

# ADIOS2
if [ -d ${SRC_DIR}/adios2 ]
then
  cd ${SRC_DIR}/adios2
  git fetch --prune
  git checkout v2.10.2
  cd -
else
  git clone -b v2.10.2 https://github.com/ornladios/ADIOS2.git ${SRC_DIR}/adios2
fi
cmake \
    --fresh                      \
    -S ${SRC_DIR}/adios2         \
    -B ${build_dir}/adios2-build \
    -DADIOS2_USE_Blosc2=ON       \
    -DADIOS2_USE_Campaign=OFF    \
    -DADIOS2_USE_Fortran=OFF     \
    -DADIOS2_USE_Python=OFF      \
    -DADIOS2_USE_ZeroMQ=OFF      \
    -DCMAKE_INSTALL_PREFIX=${SW_DIR}/adios2-2.10.2
cmake \
    --build ${build_dir}/adios2-build \
    --target install                  \
    --parallel ${build_procs}
rm -rf ${build_dir}/adios2-build

# BLAS++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/blaspp ]
then
  cd ${SRC_DIR}/blaspp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/blaspp.git ${SRC_DIR}/blaspp
fi
cmake \
    --fresh \
    -S ${SRC_DIR}/blaspp                      \
    -B ${build_dir}/blaspp-tioga-mi300a-build \
    -Duse_openmp=OFF                          \
    -Dgpu_backend=hip                         \
    -DCMAKE_CXX_STANDARD=17                   \
    -DCMAKE_INSTALL_PREFIX=${SW_DIR}/blaspp-2024.05.31
cmake \
    --build ${build_dir}/blaspp-tioga-mi300a-build \
    --target install                               \
    --parallel ${build_procs}
rm -rf ${build_dir}/blaspp-tioga-mi300a-build

# LAPACK++ (for PSATD+RZ)
if [ -d ${SRC_DIR}/lapackpp ]
then
  cd ${SRC_DIR}/lapackpp
  git fetch --prune
  git checkout v2024.05.31
  cd -
else
  git clone -b v2024.05.31 https://github.com/icl-utk-edu/lapackpp.git ${SRC_DIR}/lapackpp
fi
cmake \
    --fresh                                     \
    -S ${SRC_DIR}/lapackpp                      \
    -B ${build_dir}/lapackpp-tioga-mi300a-build \
    -DCMAKE_CXX_STANDARD=17                     \
    -Dgpu_backend=hip                           \
    -Dbuild_tests=OFF                           \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON      \
    -DCMAKE_INSTALL_PREFIX=${SW_DIR}/lapackpp-2024.05.31
cmake \
    --build ${build_dir}/lapackpp-tioga-mi300a-build \
    --target install                                 \
    --parallel ${build_procs}
rm -rf ${build_dir}/lapackpp-tioga-mi300a-build

# Python ######################################################################
#
# sometimes, the Lassen PIP Index is down
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
# python3 -m pip cache purge || true  # Cache disabled on system
rm -rf ${SW_DIR}/venvs/warpx-trioga-mi300a
python3 -m venv ${SW_DIR}/venvs/warpx-trioga-mi300a
source ${SW_DIR}/venvs/warpx-trioga-mi300a/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade openpmd-viewer
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
# install or update WarpX dependencies such as picmistandard
python3 -m pip install --upgrade -r ${SRC_DIR}/warpx/requirements.txt
# cupy for ROCm
#   https://docs.cupy.dev/en/stable/install.html#building-cupy-for-rocm-from-source
#   https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental
#   https://github.com/cupy/cupy/issues/7830
#   https://github.com/cupy/cupy/pull/8457
#   https://github.com/cupy/cupy/pull/8319
#python3 -m pip install --upgrade "cython<3"
#HIPCC=${CXX} \
#CXXFLAGS="-I${ROCM_PATH}/include/hipblas -I${ROCM_PATH}/include/hipsparse -I${ROCM_PATH}/include/hipfft -I${ROCM_PATH}/include/rocsolver -I${ROCM_PATH}/include/rccl" \
#CUPY_INSTALL_USE_HIP=1  \
#ROCM_HOME=${ROCM_PATH}  \
#HCC_AMDGPU_TARGET=${AMREX_AMD_ARCH}  \
#  python3 -m pip install -v cupy
#python3 -m pip install --upgrade "cython>=3"


# for ML dependencies, see install_mi300a_ml.sh

# remove build temporary directory
rm -rf ${build_dir}
