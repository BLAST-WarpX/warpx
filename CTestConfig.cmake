## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
##
## # The following are required to submit to the CDash dashboard:
##   ENABLE_TESTING()
##   INCLUDE(CTest)

set(CTEST_PROJECT_NAME WarpX)
set(CTEST_NIGHTLY_START_TIME 08:00:00 UTC)

set(CTEST_SUBMIT_URL https://my.cdash.org/submit.php?project=WarpX)

set(CTEST_DROP_SITE_CDASH TRUE)

# Set site and build names
# - CTest script variables: CTEST_SITE, CTEST_BUILD_NAME
# - CTest module variables: SITE, BUILDNAME
set(SITE "Azure-Pipelines")
set(BUILDNAME "CI-Development")
