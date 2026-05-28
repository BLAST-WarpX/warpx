.. _developers-release:

Dependencies & Releases
=======================

Update WarpX' Core Dependencies
-------------------------------

WarpX has direct dependencies on AMReX, pyAMReX, and PICSAR, which we periodically update.

The script ``update_dependencies.py`` from `Tools/Release/update_dependencies.py <https://github.com/BLAST-WarpX/warpx/blob/development/Tools/Release/update_dependencies.py>`__ automates this workflow, in case one needs a newer commit of AMReX, pyAMReX, or PICSAR between releases:

.. code-block:: bash

     usage: update_dependencies.py [-h] [--amrex] [--pyamrex] [--picsar] [--warpx]
     options:
       -h, --help  show this help message and exit
       --amrex     Update AMReX only
       --pyamrex   Update pyAMReX only
       --picsar    Update PICSAR only
       --warpx     Update WarpX only
       --release   New release

Create a new WarpX release
--------------------------

WarpX has one release per month.
The version number is set at the beginning of the month and follows the format ``YY.MM``.

In order to create a GitHub release, you need to:

 1. Create a new branch from ``development`` and update the version number in all source files.
    We usually wait for the AMReX release to be tagged first, then we also point to its tag.

    The script above can be used to update the core dependencies of WarpX and the WarpX version.

    For a WarpX release, ideally a *git tag* of AMReX & PICSAR shall be used instead of an unnamed commit.
    This can be done by running the script above with the command line option ``--release``.

    Then open a PR, wait for tests to pass and then merge.

    The maintainer script ``Tools/Release/releasePR.py`` automates the steps above.
    Please read through the instructions in the script before running.

 2. **Local Commit** (Optional): at the moment, ``@ax3l`` is managing releases and signs tags (naming: ``YY.MM``) locally with his GPG key before uploading them to GitHub.

    **Publish**: On the `GitHub Release page <https://github.com/BLAST-WarpX/warpx/releases>`__, create a new release via ``Draft a new release``.
    Either select the locally created tag or create one online (naming: ``YY.MM``) on the merged commit of the PR from step 1.

    In the *release description*, please specify the compatible versions of dependencies (see previous releases), and provide info on the content of the release.
    In order to get a list of PRs merged since last release, you may run

    .. code-block:: sh

       git log <last-release-tag>.. --format='- %s'

 3. Optional/future: create a ``release-<version>`` branch, write a changelog, and backport bug-fixes for a few days.
