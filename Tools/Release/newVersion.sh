#!/usr/bin/env bash
#
# Copyright 2021 Axel Huebl
#
# This file is part of WarpX.
#

# This file is a maintainer tool to bump the versions inside WarpX'
# source directory at all places where necessary.
#
# Note: this script is only tested with GNUtools (Linux)

set -eu -o pipefail


# Maintainer Inputs ###########################################################

echo "Hi there, this is a WarpX maintainer tool to update the source"
echo "code of WarpX to a new version number on all places where"
echo "necessary."
echo "For it to work, you need write access on the source directory and"
echo "you should be working in a clean git branch without ongoing"
echo "rebase/merge/conflict resolves and without unstaged changes."

# check source dir
REPO_DIR=$(cd $(dirname ${BASH_SOURCE})/../../ && pwd)
echo
echo "Your current source directory is: ${REPO_DIR}"
echo

read -p "Are you sure you want to continue? [y/N] " -r
echo

if [[ ! ${REPLY} =~ ^[Yy]$ ]]
then
    echo "You did not confirm with 'y', aborting."
    exit 1
fi

echo "We will now run a few sed commands on your source directory."
echo "Please answer the following questions about the version number"
echo "you want to set first:"
echo

read -p "MAJOR version? (e.g. year: $(date +%y)) " -r
MAJOR=${REPLY}
echo
read -p "MINOR version? (e.g. month: $(date +%m)) " -r
MINOR=${REPLY}
echo
read -p "PATCH version? (e.g. usually empty) " -r
PATCH=${REPLY}
echo
read -p "SUFFIX? (e.g. rc2, dev, ... usually empty) " -r
SUFFIX=${REPLY}
echo

if [[ -n "${SUFFIX}" ]]
then
    SUFFIX_STR="-$SUFFIX"
else
    SUFFIX_STR=""
fi
if [[ ! -n "${PATCH}" ]]
then
    PATCH=""
fi

VERSION_STR_NOSUFFIX="${MAJOR}.${MINOR}${PATCH}"
VERSION_STR="${MAJOR}.${MINOR}${PATCH}${SUFFIX_STR}"

echo
echo "Your new version is: ${VERSION_STR}"
echo

read -p "Is this information correct? Will now start updating! [y/N] " -r
echo

if [[ ! ${REPLY} =~ ^[Yy]$ ]]
then
    echo "You did not confirm with 'y', aborting."
    exit 1
fi


# Updates #####################################################################

# CMake scripts
#   CMakeLists.txt: project(WarpX VERSION YY.MM)
sed -i -E "s/"\
"(project\(WarpX VERSION[[:blank:]]+)(.*)(\))/"\
"\1${VERSION_STR_NOSUFFIX}\3/g" \
    ${REPO_DIR}/CMakeLists.txt

#   cmake/dependencies/AMReX.cmake:
#     set(WarpX_amrex_branch "development" ... (future)
#     find_package(AMReX YY.MM CONFIG ...
sed -i -E "s/"\
"(find_package\(AMReX[[:blank:]]+)(.*)([[:blank:]]+CONFIG.+)/"\
"\1${VERSION_STR_NOSUFFIX}\3/g" \
    ${REPO_DIR}/cmake/dependencies/AMReX.cmake

#   cmake/dependencies/PICSAR.cmake (future)

# setup.py: version = '21.02',
sed -i -E "s/"\
"([[:blank:]]*version[[:blank:]]*=[[:blank:]]*\")(.*)(\".+)/"\
"\1${VERSION_STR}\3/g" \
    ${REPO_DIR}/setup.py

# Python/setup.py: version = '21.02',
sed -i -E "s/"\
"([[:blank:]]*version[[:blank:]]*=[[:blank:]]*\")(.*)(\".+)/"\
"\1${VERSION_STR}\3/g" \
    ${REPO_DIR}/Python/setup.py

# sphinx / RTD
#   docs/source/conf.py
sed -i "s/"\
"[[:blank:]]*version[[:blank:]]*=[[:blank:]]*.*/"\
"version = \"${VERSION_STR_NOSUFFIX}\"/g" \
    ${REPO_DIR}/Docs/source/conf.py
sed -i "s/"\
"[[:blank:]]*release[[:blank:]]*=[[:blank:]]*.*/"\
"release = \"${VERSION_STR}\"/g" \
    ${REPO_DIR}/Docs/source/conf.py


# Epilog ######################################################################

echo
echo "Done. Please check your source, e.g. via"
echo "  git diff"
echo "now and commit the changes if no errors occured."
