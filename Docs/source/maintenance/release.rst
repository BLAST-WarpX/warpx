.. _developers-release:

Dependencies & Releases
=======================

Update WarpX' Core Dependencies
-------------------------------

WarpX has direct dependencies on AMReX and PICSAR, which we periodically update.

The following scripts automate this workflow, in case one needs a newer commit of AMReX or PICSAR between releases:

.. code-block:: sh

   ./Tools/Release/updateAMReX.py
   ./Tools/Release/updatepyAMReX.py
   ./Tools/Release/updatePICSAR.py


Create a new WarpX release
--------------------------

WarpX has one release per month.
The version number is set at the beginning of the month and follows the format ``YY.MM``.

In order to create a GitHub release, you need to:

 1. Create a new branch from ``development`` and update the version number in all source files.
    We usually wait for the AMReX release to be tagged first, then we also point to its tag.

    There are scripts for updating core dependencies of WarpX and the WarpX version:

    .. code-block:: sh

       ./Tools/Release/updateAMReX.py
       ./Tools/Release/updatepyAMReX.py
       ./Tools/Release/updatePICSAR.py

       ./Tools/Release/newVersion.sh

    For a WarpX release, ideally a *git tag* of AMReX & PICSAR shall be used instead of an unnamed commit.

    Then open a PR, wait for tests to pass and then merge.

    The maintainer script ``Tools/Release/releasePR.py`` automates the steps above.
    Please read through the instructions in the script before running.

 2. **Local Commit** (Optional): at the moment, ``@ax3l`` is managing releases and signs tags (naming: ``YY.MM``) locally with his GPG key before uploading them to GitHub.

    **Publish**: On the `GitHub Release page <https://github.com/ECP-WarpX/WarpX/releases>`__, create a new release via ``Draft a new release``.
    Either select the locally created tag or create one online (naming: ``YY.MM``) on the merged commit of the PR from step 1.

    In the *release description*, please specify the compatible versions of dependencies (see previous releases), and provide info on the content of the release.
    In order to get a list of PRs merged since last release, you may run

    .. code-block:: sh

       git log <last-release-tag>.. --format='- %s'

 3. Optional/future: create a ``release-<version>`` branch, write a changelog, and backport bug-fixes for a few days.
