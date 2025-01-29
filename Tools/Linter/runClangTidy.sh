#!/usr/bin/env bash
#
# Copyright 2024 Luca Fedeli
#
# This file is part of WarpX.
#

# This script is a developer's tool to perform the
# checks done by the clang-tidy CI test locally.
#
# Note: this script is only tested on Linux

echo "============================================="
echo
echo "This script is a developer's tool to perform the"
echo "checks done by the clang-tidy CI test locally"
echo "_____________________________________________"

# Check source dir
REPO_DIR=$(cd $(dirname ${BASH_SOURCE})/../../ && pwd)
echo
echo "Your current source directory is: ${REPO_DIR}"
echo "_____________________________________________"

# Set number of jobs to use for compilation
PARALLEL="${WARPX_TOOLS_LINTER_PARALLEL:-4}"
echo
echo "${PARALLEL} jobs will be used for compilation."
echo "This can be overridden by setting the environment"
echo "variable WARPX_TOOLS_LINTER_PARALLEL, e.g.: "
echo
echo "$ export WARPX_TOOLS_LINTER_PARALLEL=8"
echo "$ ./Tools/Linter/runClangTidy.sh"
echo "_____________________________________________"

# Check clang version
export CC="${CLANG:-"clang"}"
export CXX="${CLANGXX:-"clang++"}"
export CTIDY="${CLANGTIDY:-"clang-tidy"}"
echo
echo "The following versions of the clang compiler and"
echo "of the clang-tidy linter will be used:"
echo
echo "clang version:"
which ${CC}
${CC} --version
echo
echo "clang++ version:"
which ${CXX}
${CXX} --version
echo
echo "clang-tidy version:"
which ${CTIDY}
${CTIDY} --version
echo
echo "This can be overridden by setting the environment"
echo "variables CLANG, CLANGXX, and CLANGTIDY e.g.: "
echo "$ export CLANG=clang-15"
echo "$ export CLANGXX=clang++-15"
echo "$ export CTIDCLANGTIDYY=clang-tidy-15"
echo "$ ./Tools/Linter/runClangTidy.sh"
echo
echo "******************************************************"
echo "* Warning: clang v15 is currently used in CI tests.  *"
echo "* It is therefore recommended to use this version.   *"
echo "* Otherwise, a newer version may find issues not     *"
echo "* currently covered by CI tests while older versions *"
echo "* may not find all the issues.                       *"
echo "******************************************************"
echo "_____________________________________________"

# Prepare clang-tidy wrapper
echo
echo "Prepare clang-tidy wrapper"
echo "The following wrapper ensures that only source files"
echo "in WarpX/Source/* are actually processed by clang-tidy"
echo
cat > ${REPO_DIR}/clang_tidy_wrapper << EOF
#!/bin/bash
REGEX="[a-z_A-Z0-9\/]*WarpX\/Source[a-z_A-Z0-9\/]+.cpp"
if [[ \$4 =~ \$REGEX ]];then
  ${CTIDY} \$@
fi
EOF
chmod +x ${REPO_DIR}/clang_tidy_wrapper
echo "clang_tidy_wrapper: "
cat ${REPO_DIR}/clang_tidy_wrapper
echo "_____________________________________________"

# Compile Warpx using clang-tidy
echo
echo "*******************************************"
echo "* Compile Warpx using clang-tidy          *"
echo "* Please ensure that all the dependencies *"
echo "* required to compile WarpX are met       *"
echo "*******************************************"
echo

rm -rf ${REPO_DIR}/build_clang_tidy

cmake -S ${REPO_DIR} -B ${REPO_DIR}/build_clang_tidy \
  -DCMAKE_CXX_CLANG_TIDY="${REPO_DIR}/clang_tidy_wrapper;--system-headers=0;--config-file=${REPO_DIR}/.clang-tidy" \
  -DCMAKE_VERBOSE_MAKEFILE=ON  \
  -DWarpX_DIMS="1;2;3;RZ"      \
  -DWarpX_MPI=ON               \
  -DWarpX_COMPUTE=OMP          \
  -DWarpX_FFT=ON               \
  -DWarpX_QED=ON               \
  -DWarpX_QED_TABLE_GEN=ON     \
  -DWarpX_OPENPMD=ON           \
  -DWarpX_PRECISION=SINGLE

cmake --build ${REPO_DIR}/build_clang_tidy -j ${PARALLEL} 2> ${REPO_DIR}/build_clang_tidy/clang-tidy.log

cat ${REPO_DIR}/build_clang_tidy/clang-tidy.log
echo
echo "============================================="
