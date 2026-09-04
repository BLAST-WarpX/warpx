# CTest driver script for CDash submissions from CI.
#
# CDash only reports configure and build results -- timings, compiler warnings
# and errors -- if those steps run *through* CTest, which records them in
# Configure.xml and Build.xml.  Calling `cmake` and `cmake --build` directly and
# then running `ctest -D ExperimentalTest` produces a Test.xml only, which is
# why our dashboard used to show test results and nothing else.
#
# This script runs the configure and build steps via ctest_configure() and
# ctest_build().  The test step is deliberately left to the caller, so that CI
# keeps using the plain `ctest` command line it already has (including options
# such as `--no-tests=error`, which ctest_test() does not offer).  Every step
# writes into the same dashboard tag created by ctest_start() here, so a single
# `ctest -D ExperimentalSubmit` at the end uploads all parts as one build.
#
# Do not invoke this file directly, use Tools/CI/ctest_dashboard.sh.
#
# Environment:
#   CDASH_SOURCE_DIR    top-level WarpX source directory (required)
#   CDASH_BINARY_DIR    build directory (required)
#   CDASH_BUILD_NAME    dashboard build name, e.g. "CPU-3D" (required)
#   CDASH_SITE          dashboard site name, e.g. "Azure" (required)
#   CDASH_OPTIONS_FILE  file with one CMake configure option per line
#   CDASH_BUILD_TARGET  build this target instead of the default one
#   CDASH_SUBMIT        "ON" to submit right after the build (build-only jobs)
#   CMAKE_GENERATOR     CMake generator, unless given as -G in the options
#   CMAKE_BUILD_PARALLEL_LEVEL  build parallelism, passed on to ctest_build()

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS CDASH_SOURCE_DIR CDASH_BINARY_DIR CDASH_BUILD_NAME CDASH_SITE)
    if("$ENV{${_required}}" STREQUAL "")
        message(FATAL_ERROR "ctest_dashboard: environment variable ${_required} is not set")
    endif()
endforeach()

set(CTEST_SOURCE_DIRECTORY "$ENV{CDASH_SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY "$ENV{CDASH_BINARY_DIR}")
set(CTEST_BUILD_NAME       "$ENV{CDASH_BUILD_NAME}")
set(CTEST_SITE             "$ENV{CDASH_SITE}")

# Configure options are handed over one per line in a file, so that values
# containing spaces (-DCMAKE_CXX_FLAGS="-Werror -Wall") or semicolons
# (-DWarpX_DIMS="1;2") survive without any shell or CMake list quoting games.
set(configure_options "")
if(NOT "$ENV{CDASH_OPTIONS_FILE}" STREQUAL "")
    # ENCODING: file(STRINGS) otherwise drops non-ASCII bytes silently
    file(STRINGS "$ENV{CDASH_OPTIONS_FILE}" raw_options ENCODING UTF-8)
    foreach(option IN LISTS raw_options)
        # escape embedded ";" so that each line stays a single list element
        string(REPLACE ";" "\;" option "${option}")
        list(APPEND configure_options "${option}")
    endforeach()
endif()

# ctest_configure() refuses to run without a generator, and appends -G after the
# options we pass. An explicit -G among them therefore has to be picked up here,
# or our own -G would silently override it. Both "-GNinja" and "-G;Ninja" occur.
set(CTEST_CMAKE_GENERATOR "$ENV{CMAKE_GENERATOR}")
set(next_is_generator FALSE)
foreach(option IN LISTS configure_options)
    if(next_is_generator)
        set(CTEST_CMAKE_GENERATOR "${option}")
        set(next_is_generator FALSE)
    elseif(option STREQUAL "-G")
        set(next_is_generator TRUE)
    elseif(option MATCHES "^-G(.+)$")
        set(CTEST_CMAKE_GENERATOR "${CMAKE_MATCH_1}")
    endif()
endforeach()
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
    if(NOT "$ENV{CDASH_BUILD_TARGET}" STREQUAL "")
        ctest_build(TARGET "$ENV{CDASH_BUILD_TARGET}"
                    ${build_parallel_arg} RETURN_VALUE build_rv)
    else()
        ctest_build(${build_parallel_arg} RETURN_VALUE build_rv)
    endif()
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
