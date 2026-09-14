# CTest driver script for CDash submissions from CI.
#
# CDash only reports configure and build results -- timings, compiler warnings
# and errors -- if those steps run *through* CTest, which records them in
# Configure.xml and Build.xml.  Calling `cmake` and `cmake --build` directly and
# then running `ctest -D ExperimentalTest` produces a Test.xml only, which is
# why our dashboard used to show test results and nothing else.
#
# The command line route (`ctest -D ExperimentalConfigure`) cannot do this: it
# drives an already-generated build tree and needs a DartConfiguration.tcl, which
# include(CTest) only writes during a configure.  It could therefore only time a
# *re*-configure, which for WarpX -- where the cold configure fetches AMReX,
# pyAMReX and PICSAR -- understates the real cost by minutes.  Driving the
# configure from a dashboard client script, as here, is the documented way.
#
# The test step is deliberately left to the caller, so that CI keeps using the
# plain `ctest` command line it already has, including options such as
# `--no-tests=error` which ctest_test() does not offer.  Every step writes into
# the same dashboard tag created by ctest_start() here, so a single
# `ctest -D ExperimentalSubmit` at the end uploads all parts as one build.
#
# Do not invoke this file directly, use Tools/CI/ctest_dashboard.sh.
#
# Environment:
#   CDASH_BUILD_DIR   build directory, absolute (required)
#   CDASH_CMAKE_ARGS  configure options, shell-quoted by the wrapper
#   CDASH_BUILD_NAME  dashboard build name, e.g. "CPU-3D" (required)
#   CDASH_SITE        dashboard site name, e.g. "Azure" (required)
#   CDASH_SUBMIT      "ON" to submit right after the build (build-only jobs)
#   CMAKE_GENERATOR   CMake generator
#   CMAKE_BUILD_PARALLEL_LEVEL  build parallelism, passed to ctest_build()

cmake_minimum_required(VERSION 3.25)

foreach(required IN ITEMS CDASH_BUILD_DIR CDASH_BUILD_NAME CDASH_SITE)
    if("$ENV{${required}}" STREQUAL "")
        message(FATAL_ERROR "ctest_dashboard: environment variable ${required} is not set")
    endif()
endforeach()

get_filename_component(CTEST_SOURCE_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}/../.." ABSOLUTE)
set(CTEST_BINARY_DIRECTORY "$ENV{CDASH_BUILD_DIR}")
set(CTEST_BUILD_NAME       "$ENV{CDASH_BUILD_NAME}")
set(CTEST_SITE             "$ENV{CDASH_SITE}")

# The wrapper hands the options over shell-quoted; UNIX_COMMAND parses that back
# into one list element per argument and escapes any embedded ";" on the way, so
# values with spaces (-DCMAKE_CXX_FLAGS="-Werror -Wall") and semicolons
# (-DWarpX_DIMS="1;2") reach cmake exactly as CI wrote them.
separate_arguments(configure_options UNIX_COMMAND "$ENV{CDASH_CMAKE_ARGS}")

# BUILDNAME and SITE are what include(CTest) writes into DartConfiguration.tcl,
# from where the caller's `ctest -D ExperimentalTest` picks them up for Test.xml.
# Without them that step would label Test.xml with the host name and a generic
# build name, and CDash would file it as a build separate from Configure.xml and
# Build.xml.
list(APPEND configure_options
     "-DBUILDNAME=${CTEST_BUILD_NAME}" "-DSITE=${CTEST_SITE}")

# ctest_configure() appends its own -G *after* these options, so one passed here
# would be silently overridden. Use the environment variable, which cmake reads
# natively, and say so rather than building the wrong thing.
if("${configure_options}" MATCHES "(^|;)-G")
    message(FATAL_ERROR
            "ctest_dashboard: pass the generator in CMAKE_GENERATOR, not as -G")
endif()
set(CTEST_CMAKE_GENERATOR "$ENV{CMAKE_GENERATOR}")
if(CTEST_CMAKE_GENERATOR STREQUAL "")
    # same default as a plain `cmake` call on the platforms we submit from
    set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
endif()

# Pass the parallelism to ctest_build() explicitly rather than relying on it
# leaking through the environment into `cmake --build`. When it is unset, pass
# nothing at all: forcing a level here would drop Ninja from its own default of
# one job per core down to whatever we picked.
set(build_parallel_arg "")
if("$ENV{CMAKE_BUILD_PARALLEL_LEVEL}" MATCHES "^[1-9][0-9]*$")
    set(build_parallel_arg PARALLEL_LEVEL "$ENV{CMAKE_BUILD_PARALLEL_LEVEL}")
endif()

# ctest_start() reuses an existing Testing/TAG only while its date still matches
# today's in UTC. A job that configures before and tests after midnight UTC
# therefore splits into two dashboard builds; rare, and it heals on the next run.
ctest_start(Experimental)

ctest_configure(OPTIONS "${configure_options}" RETURN_VALUE configure_rv)

# A failed configure leaves nothing to build.
set(build_rv 0)
if(configure_rv EQUAL 0)
    # Only ever the default target: CTest finds build errors by matching regexes
    # against the build log, so a packaging target that shells out to pip would
    # have its setuptools warnings reported as build errors.
    ctest_build(${build_parallel_arg} RETURN_VALUE build_rv)
endif()

# Submit when the caller asked for it, and *always* when the configure failed.
# A caller that submits from its own `ctest -D ExperimentalSubmit` step cannot
# do it in that case: that command reads the build directory through
# DartConfiguration.tcl, which include(CTest) only writes once the configure has
# got that far, so it would abort and discard the Configure.xml holding the
# CMake error. Here the submit is driven by CTestConfig.cmake in the source
# directory instead, and works no matter how early the configure died.
if("$ENV{CDASH_SUBMIT}" OR NOT configure_rv EQUAL 0)
    # Never let the dashboard decide whether the build passed: a CDash outage
    # must not turn a green build red. RETURN_VALUE alone does not do that --
    # CAPTURE_CMAKE_ERROR is what keeps ctest from exiting non-zero.
    ctest_submit(RETRY_COUNT 3 RETRY_DELAY 15
                 RETURN_VALUE submit_rv CAPTURE_CMAKE_ERROR submit_err)
    if(NOT submit_rv EQUAL 0 OR NOT submit_err EQUAL 0)
        message("ctest_dashboard: submission to CDash failed, continuing anyway")
    endif()
endif()

if(NOT configure_rv EQUAL 0)
    message(FATAL_ERROR "ctest_dashboard: configure failed")
endif()
if(NOT build_rv EQUAL 0)
    message(FATAL_ERROR "ctest_dashboard: build failed")
endif()
